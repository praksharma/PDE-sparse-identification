// -----------------------------------------------------------------------------
//
//  Gmsh C++ tutorial 13
//
//  Remeshing an STL file without an underlying CAD model
//
// -----------------------------------------------------------------------------

#include <set>
#include <cmath>
#include <gmsh.h>

int main(int argc, char **argv)
{
  gmsh::initialize();

  auto createGeometryAndMesh = []()
  {
    gmsh::model::add("t13");

    // Let's merge an STL mesh that we would like to remesh (from the parent
    // directory):
    try {
      gmsh::merge("../t13_data.stl");
    } catch(...) {
      gmsh::logger::write("Could not load STL mesh: bye!");
      return;
    }

    // We first classify ("color") the surfaces by splitting the original
    // surface along sharp geometrical features. This will create new discrete
    // surfaces, curves and points.

    // Angle between two triangles above which an edge is considered as sharp,
    // retrieved from the ONELAB database (see below):
    std::vector<double> n;
    gmsh::onelab::getNumber("Parameters/Angle for surface detection", n);
    double angle = n[0];

    // For complex geometries, patches can be too complex, too elongated or too
    // large to be parametrized; setting the following option will force the
    // creation of patches that are amenable to reparametrization:
    gmsh::onelab::getNumber
      ("Parameters/Create surfaces guaranteed to be parametrizable", n);
    bool forceParametrizablePatches = n[0] ? true : false;

    // For open surfaces include the boundary edges in the classification process:
    bool includeBoundary = true;

    // Force curves to be split on given angle:
    double curveAngle = 180;

    gmsh::model::mesh::classifySurfaces(angle * M_PI / 180., includeBoundary,
                                        forceParametrizablePatches,
                                        curveAngle * M_PI / 180.);

    // Create a geometry for all the discrete curves and surfaces in the mesh,
    // by computing a parametrization for each one
    gmsh::model::mesh::createGeometry();

    // Note that if a CAD model (e.g. as a STEP file, see `t20.cpp') is
    // available instead of an STL mesh, it is usually better to use that CAD
    // model instead of the geometry created by reparametrizing the
    // mesh. Indeed, CAD geometries will in general be more accurate, with
    // smoother parametrizations, and will lead to more efficient and higher
    // quality meshing. Discrete surface remeshing in Gmsh is optimized to
    // handle dense STL meshes coming from e.g. imaging systems, where no CAD is
    // available; it is less well suited for the poor quality STL triangulations
    // (optimized for size, with e.g. very elongated triangles) that are usually
    // generated by CAD tools for e.g. 3D printing.

    // Create a volume from all the surfaces
    std::vector<std::pair<int, int> > s;
    gmsh::model::getEntities(s, 2);
    std::vector<int> sl;
    for(auto surf : s) sl.push_back(surf.second);
    int l = gmsh::model::geo::addSurfaceLoop(sl);
    gmsh::model::geo::addVolume({l});

    gmsh::model::geo::synchronize();

    // We specify element sizes imposed by a size field, just because we can :-)
    int f = gmsh::model::mesh::field::add("MathEval");
    gmsh::onelab::getNumber("Parameters/Apply funny mesh size field?", n);
    if(n[0])
      gmsh::model::mesh::field::setString(f, "F", "2*Sin((x+y)/5) + 3");
    else
      gmsh::model::mesh::field::setString(f, "F", "4");
    gmsh::model::mesh::field::setAsBackgroundMesh(f);

    gmsh::model::mesh::generate(3);
  };

  // Create ONELAB parameters with remeshing options:
  gmsh::onelab::set(R"( [
  {
    "type":"number",
    "name":"Parameters/Angle for surface detection",
    "values":[40],
    "min":20,
    "max":120,
    "step":1
  },
  {
    "type":"number",
    "name":"Parameters/Create surfaces guaranteed to be parametrizable",
    "values":[0],
    "choices":[0, 1]
  },
  {
    "type":"number",
    "name":"Parameters/Apply funny mesh size field?",
    "values":[0],
    "choices":[0, 1]
  }
  ] )");

  // Create the geometry and mesh it:
  createGeometryAndMesh();

  // Launch the GUI and handle the "check" event to recreate the geometry and mesh
  // with new parameters if necessary:
  auto checkForEvent = [=]() -> bool {
    std::vector<std::string> action;
    gmsh::onelab::getString("ONELAB/Action", action);
    if(action.size() and action[0] == "check") {
      gmsh::onelab::setString("ONELAB/Action", {""});
      createGeometryAndMesh();
      gmsh::graphics::draw();
    }
    return true;
  };

  std::set<std::string> args(argv, argv + argc);
  if(!args.count("-nopopup")) {
    gmsh::fltk::initialize();
    while(gmsh::fltk::isAvailable() && checkForEvent())
      gmsh::fltk::wait();
  }

  gmsh::finalize();
  return 0;
}
