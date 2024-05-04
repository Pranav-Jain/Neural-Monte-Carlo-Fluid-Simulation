#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <zombie/point_estimation/walk_on_stars.h>
#include <zombie/boundary_value_caching/splatter.h>
#include <zombie/utils/progress.h>
#include "grid.h"
#include "scene_3d.h"
#include "pybind11_json.h"
#include <tuple>

using json = nlohmann::json;
namespace py = pybind11;

std::tuple<std::vector<std::vector<float>>, std::vector<float>, std::vector<std::vector<float>>> runWalkOnStars_3d(const Scene& scene, const json& solverConfig, const json& outputConfig, std::vector<std::vector<float>> sample_points) {
// std::tuple<std::vector<std::vector<float>>, std::vector<float>, std::vector<std::vector<float>>> runWalkOnStars_sampled(const Scene& scene, const json& solverConfig, const json& outputConfig) {
	// load configuration settings
	const bool disableGradientControlVariates = getOptional<bool>(solverConfig, "disableGradientControlVariates", false);
	const bool disableGradientAntitheticVariates = getOptional<bool>(solverConfig, "disableGradientAntitheticVariates", false);
	const bool useCosineSamplingForDirectionalDerivatives = getOptional<bool>(solverConfig, "useCosineSamplingForDirectionalDerivatives", false);
	const bool ignoreDirichlet = getOptional<bool>(solverConfig, "ignoreDirichlet", false);
	const bool ignoreNeumann = getOptional<bool>(solverConfig, "ignoreNeumann", false);
	const bool ignoreSource = getOptional<bool>(solverConfig, "ignoreSource", false);

	const int nWalks = getOptional<int>(solverConfig, "nWalks", 128);
	const int maxWalkLength = getOptional<int>(solverConfig, "maxWalkLength", 1024);
	const int stepsBeforeApplyingTikhonov = getOptional<int>(solverConfig, "setpsBeforeApplyingTikhonov", maxWalkLength);
	const int stepsBeforeUsingMaximalSpheres = getOptional<int>(solverConfig, "setpsBeforeUsingMaximalSpheres", maxWalkLength);
	const int gridRes = getRequired<int>(outputConfig, "gridRes");

	const float epsilonShell = getOptional<float>(solverConfig, "epsilonShell", 1e-3f);
	const float minStarRadius = getOptional<float>(solverConfig, "minStarRadius", 1e-3f);
	const float silhouettePrecision = getOptional<float>(solverConfig, "silhouettePrecision", 1e-3f);
	const float russianRouletteThreshold = getOptional<float>(solverConfig, "russianRouletteThreshold", 0.0f);

	fcpw::BoundingBox<3> bbox = scene.bbox;
	const zombie::GeometricQueries<3>& queries = scene.queries;
	const zombie::PDE<float, 3>& pde = scene.pde;
	bool solveDoubleSided = scene.isDoubleSided;

	// setup solution domain
	// std::vector<zombie::SamplePoint<float, 2>> samplePts;
	// createSolutionGrid(samplePts, queries, bbox.pMin, bbox.pMax, gridRes, pts);

	std::vector<zombie::SamplePoint<float, 3>> samplePts;
	// std::vector<std::vector<float>> sample_points;
	createSolutionGrid_3d(samplePts, queries, bbox.pMin, bbox.pMax, gridRes, sample_points);

	std::vector<zombie::SampleEstimationData<3>> sampleEstimationData(samplePts.size());
	for (int i = 0; i < samplePts.size(); i++) {
		sampleEstimationData[i].nWalks = nWalks;
		sampleEstimationData[i].estimationQuantity = queries.insideDomain(samplePts[i].pt) || solveDoubleSided ?
													 zombie::EstimationQuantity::SolutionAndGradient:
													 zombie::EstimationQuantity::None;
	}

	// initialize solver and estimate solution
	Vector3 extent = bbox.pMax - bbox.pMin;
	// float scale1 = 1;
	// float scale2 = 1;
	// float scale3 = 1;
	// if (extent.x() > extent.y()) {
	// 	if (extent.y() > extent.z()) {
	// 		scale1 = extent.x()/extent.z();
	// 		scale2 = extent.y()/extent.z();
	// 	}
	// 	else {
	// 		scale1 = extent.x()/extent.y();
	// 		scale3 = extent.z()/extent.y();
	// 	}
	// }
	// else {
	// 	if (extent.x() > extent.z()) {
	// 		scale1 = extent.x()/extent.z();
	// 		scale2 = extent.y()/extent.z();
	// 	}
	// 	else {
	// 		scale2 = extent.y()/extent.x();
	// 		scale3 = extent.z()/extent.x();
	// 	}
	// }
	ProgressBar pb(sample_points.size());
	std::function<void(int, int)> reportProgress = [&pb](int i, int tid) -> void { pb.report(i, tid); };

	zombie::WalkSettings<float> walkSettings(0.0f, epsilonShell, minStarRadius,
											 silhouettePrecision, russianRouletteThreshold,
											 maxWalkLength, stepsBeforeApplyingTikhonov,
											 stepsBeforeUsingMaximalSpheres, solveDoubleSided,
											 !disableGradientControlVariates,
											 !disableGradientAntitheticVariates,
											 useCosineSamplingForDirectionalDerivatives,
											 ignoreDirichlet, ignoreNeumann, ignoreSource, false);

	zombie::WalkOnStars<float, 3> walkOnStars(queries);
	walkOnStars.solve(pde, walkSettings, sampleEstimationData, samplePts, false, reportProgress);
	// pb.finish();

	// saveSolutionGrid(samplePts, pde, queries, solveDoubleSided, outputConfig);

	std::vector<float> solution;
	solution = getSolution_3d(samplePts, pde, queries, solveDoubleSided, outputConfig);
	std::vector<std::vector<float>> gradient;
	gradient = getGradient_3d(samplePts, pde, queries, solveDoubleSided, outputConfig);

	// std::vector<std::vector<float>> output;
	// output.resize(samplePts.size());
	// for (int i = 0; i < samplePts.size(); ++i) {
	// 	// output[i].resize(3);
	// 	output[i] = {samplePts[i].pt[0], samplePts[i].pt[1], solution[i]};
	// }

	// for(auto i: samplePts)
	// 	std::cout<<i.pt[0]<<" "<<i.pt[1]<<"\n";

	return std::make_tuple(sample_points, solution, gradient);
}


PYBIND11_MODULE(zombie_bindings, m) {
    m.doc() = "pybind11 WoSt"; // optional module docstring
    m.def("wost", &runWalkOnStars_3d);
	
	py::class_<Scene>(m, "Scene")
		.def(py::init<json &, std::vector<std::vector<std::vector<float>>> &>());
}