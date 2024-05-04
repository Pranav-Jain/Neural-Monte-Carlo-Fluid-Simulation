#pragma once

#include <filesystem>
#include <zombie/core/pde.h>
#include <random>
#include "config.h"
#include "colormap.h"

void writeSolution(const std::string& filename,
				   std::shared_ptr<Image<3>> solution,
				   std::shared_ptr<Image<3>> dirichletdistance,
				   std::shared_ptr<Image<3>> neumanndistance,
				   std::shared_ptr<Image<3>> indomain,
				   bool saveDebug, bool saveColormapped,
				   std::string colormap, float minVal, float maxVal) {
	std::filesystem::path path(filename);
	std::filesystem::create_directories(path.parent_path());

	solution->write(filename);

	std::string basePath = (path.parent_path() / path.stem()).string();
	std::string ext = path.extension();

	if (saveColormapped) {
		getColormappedImage(solution, colormap, minVal, maxVal)->write(basePath + "_color" + ext);
	}

	if (saveDebug) {
		dirichletdistance->write(basePath + "_dirichlet" + ext);
		neumanndistance->write(basePath + "_neumann" + ext);
		indomain->write(basePath + "_indomain" + ext);
	}
}

void createSolutionGrid(std::vector<zombie::SamplePoint<float, 2>>& samplePts,
						const zombie::GeometricQueries<2> &queries,
						const Vector2 &bMin, const Vector2 &bMax,
						const int gridRes) {
	Vector2 extent = bMax - bMin;
	for (int i = 0; i < gridRes; i++) {
		for (int j = 0; j < gridRes; j++) {
			Vector2 pt((i / float(gridRes)) * extent.x() + bMin.x(),
					   (j / float(gridRes)) * extent.y() + bMin.y());
			float dDist = queries.computeDistToDirichlet(pt, false);
			float nDist = queries.computeDistToNeumann(pt, false);
			samplePts.emplace_back(zombie::SamplePoint<float, 2>(pt, Vector2::Zero(),
																 zombie::SampleType::InDomain,
																 1.0f, dDist, nDist, 0.0f));
		}
	}
}

// void createSolutionGrid(std::vector<zombie::SamplePoint<float, 2>>& samplePts,
// 						const zombie::GeometricQueries<2> &queries,
// 						const Vector2 &bMin, const Vector2 &bMax,
// 						const int gridRes,
// 						std::vector<std::vector<float>> &pts) {
// 	// Vector2 extent = bMax - bMin;
// 	for (int i = 0; i < pts.size(); i++) {
// 		Vector2 pt(pts[i][0], pts[i][1]);
// 		float dDist = queries.computeDistToDirichlet(pt, false);
// 		float nDist = queries.computeDistToNeumann(pt, false);
// 		samplePts.emplace_back(zombie::SamplePoint<float, 2>(pt, Vector2::Zero(),
// 																zombie::SampleType::InDomain,
// 																1.0f, dDist, nDist, 0.0f));
// 	}
// }

void createSolutionGrid(std::vector<zombie::SamplePoint<float, 2>>& samplePts,
						const zombie::GeometricQueries<2> &queries,
						const Vector2 &bMin, const Vector2 &bMax,
						const int gridRes,
						std::vector<std::vector<float>>& pts) {
	// std::random_device rand_dev;
    // std::mt19937 generator(rand_dev());
	// std::uniform_real_distribution<float>  distr(0.0, 1.0);

	Vector2 extent = bMax - bMin;
	float scale1 = 1;
	float scale2 = 1;
	if (extent.x() > extent.y()) {
		scale1 = extent.x()/extent.y();
	}
	else {
		scale2 = extent.y()/extent.x();
	}
	for (int i = 0; i < scale1 * gridRes; i++) {
		for (int j = 0; j < scale2 * gridRes; j++) {
			Vector2 pt((i / float(scale1 * gridRes)) * extent.x() + bMin.x(),
					   (j / float(scale2 * gridRes)) * extent.y() + bMin.y());
			std::vector<float> temp {pt[0], pt[1]};
			pts.emplace_back(temp);
	// for (int i = 0; i < pts.size(); ++i) {
	// 	Vector2 pt(pts[i][0], pts[i][1]);
		
			float dDist = queries.computeDistToDirichlet(pt, false);
			float nDist = queries.computeDistToNeumann(pt, false);
			samplePts.emplace_back(zombie::SamplePoint<float, 2>(pt, Vector2::Zero(),
																	zombie::SampleType::InDomain,
																	1.0f, dDist, nDist, 0.0f));
		}
	}
}

void createSolutionGrid_3d(std::vector<zombie::SamplePoint<float, 3>>& samplePts,
						const zombie::GeometricQueries<3> &queries,
						const Vector3 &bMin, const Vector3 &bMax,
						const int gridRes,
						std::vector<std::vector<float>>& pts) {
	// std::random_device rand_dev;
    // std::mt19937 generator(rand_dev());
	// std::uniform_real_distribution<float>  distr(0.0, 1.0);

	// Vector3 extent = bMax - bMin;
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
	for (int i = 0; i < pts.size(); i++) {
		Vector3 pt(pts[i][0], pts[i][1], pts[i][2]);
		float dDist = queries.computeDistToDirichlet(pt, false);
		float nDist = queries.computeDistToNeumann(pt, false);
		samplePts.emplace_back(zombie::SamplePoint<float, 3>(pt, Vector3::Zero(),
															zombie::SampleType::InDomain,
															1.0f, dDist, nDist, 0.0f));
	}
}

std::vector<float> getSolution(const std::vector<zombie::SamplePoint<float, 2>>& samplePts,
								const zombie::PDE<float, 2>& pde,
								const zombie::GeometricQueries<2> &queries,
								const bool isDoubleSided, const json &config) {
	const float boundaryDistanceMask = getOptional<float>(config, "boundaryDistanceMask", 0.0);

	std::vector<float> solution;
	solution.resize(samplePts.size());
	for (int idx = 0; idx < samplePts.size(); idx++) {

		// debug / scene data
		float inDomain  = queries.insideDomain(samplePts[idx].pt) ? 1 : 0;
		float dirichletDist = samplePts[idx].dirichletDist;
		float neumannDist = samplePts[idx].neumannDist;

		// solution data
		float value = samplePts[idx].statistics ? samplePts[idx].statistics->getEstimatedSolution(): 0.0f;
		// bool maskOutValue = (!inDomain && !isDoubleSided) ||
		// 					std::abs(neumannDist) < boundaryDistanceMask;
		bool maskOutValue = std::abs(neumannDist) < boundaryDistanceMask;
		solution[idx] = maskOutValue ? 0.0f : value;
	}
	
	return solution;
}

std::vector<float> getSolution_3d(const std::vector<zombie::SamplePoint<float, 3>>& samplePts,
								const zombie::PDE<float, 3>& pde,
								const zombie::GeometricQueries<3> &queries,
								const bool isDoubleSided, const json &config) {
	const float boundaryDistanceMask = getOptional<float>(config, "boundaryDistanceMask", 0.0);

	std::vector<float> solution;
	solution.resize(samplePts.size());
	for (int idx = 0; idx < samplePts.size(); idx++) {

		// debug / scene data
		float inDomain  = queries.insideDomain(samplePts[idx].pt) ? 1 : 0;
		float dirichletDist = samplePts[idx].dirichletDist;
		float neumannDist = samplePts[idx].neumannDist;

		// solution data
		float value = samplePts[idx].statistics ? samplePts[idx].statistics->getEstimatedSolution(): 0.0f;
		// bool maskOutValue = (!inDomain && !isDoubleSided) ||
		// 					std::abs(neumannDist) < boundaryDistanceMask;
		bool maskOutValue = std::abs(neumannDist) < boundaryDistanceMask;
		solution[idx] = maskOutValue ? 0.0f : value;
	}
	
	return solution;
}

std::vector<std::vector<float>> getGradient(const std::vector<zombie::SamplePoint<float, 2>>& samplePts,
								const zombie::PDE<float, 2>& pde,
								const zombie::GeometricQueries<2> &queries,
								const bool isDoubleSided, const json &config) {
	const float boundaryDistanceMask = getOptional<float>(config, "boundaryDistanceMask", 0.0);

	std::vector<std::vector<float>> solution;
	solution.resize(samplePts.size());
	for (int idx = 0; idx < samplePts.size(); idx++) {
		solution[idx].resize(2);

		// debug / scene data
		float inDomain  = queries.insideDomain(samplePts[idx].pt) ? 1 : 0;
		float dirichletDist = samplePts[idx].dirichletDist;
		float neumannDist = samplePts[idx].neumannDist;

		// solution data
		std::vector<float> zero{0.0f, 0.0f};
		
		const float *value = samplePts[idx].statistics ? samplePts[idx].statistics->getEstimatedGradient() : &zero[0];
		bool maskOutValue = (!inDomain && !isDoubleSided) ||
							std::abs(neumannDist) < boundaryDistanceMask;
		
		solution[idx][0] = maskOutValue ? 0.0f : value[0];
		solution[idx][1] = maskOutValue ? 0.0f : value[1];
		// solution[idx][0] = value[0];
		// solution[idx][1] = value[1];
	}
	
	return solution;
}

std::vector<std::vector<float>> getGradient_3d(const std::vector<zombie::SamplePoint<float, 3>>& samplePts,
								const zombie::PDE<float, 3>& pde,
								const zombie::GeometricQueries<3> &queries,
								const bool isDoubleSided, const json &config) {
	const float boundaryDistanceMask = getOptional<float>(config, "boundaryDistanceMask", 0.0);

	std::vector<std::vector<float>> solution;
	solution.resize(samplePts.size());
	for (int idx = 0; idx < samplePts.size(); idx++) {
		solution[idx].resize(3);

		// debug / scene data
		float inDomain  = queries.insideDomain(samplePts[idx].pt) ? 1 : 0;
		float dirichletDist = samplePts[idx].dirichletDist;
		float neumannDist = samplePts[idx].neumannDist;

		// solution data
		std::vector<float> zero{0.0f, 0.0f, 0.0f};
		
		const float *value = samplePts[idx].statistics ? samplePts[idx].statistics->getEstimatedGradient() : &zero[0];
		bool maskOutValue = (!inDomain && !isDoubleSided) ||
							std::abs(neumannDist) < boundaryDistanceMask;
		
		solution[idx][0] = maskOutValue ? 0.0f : value[0];
		solution[idx][1] = maskOutValue ? 0.0f : value[1];
		solution[idx][2] = maskOutValue ? 0.0f : value[2];
		// solution[idx][0] = value[0];
		// solution[idx][1] = value[1];
	}
	
	return solution;
}

void saveSolutionGrid(const std::vector<zombie::SamplePoint<float, 2>>& samplePts,
					  const zombie::PDE<float, 2>& pde,
					  const zombie::GeometricQueries<2> &queries,
					  const bool isDoubleSided, const json &config) {
	const std::string solutionFile = getOptional<std::string>(config, "solutionFile", "solution.pfm");
	const int gridRes = getRequired<int>(config, "gridRes");
	const float boundaryDistanceMask = getOptional<float>(config, "boundaryDistanceMask", 0.0);

	const bool saveDebug = getOptional<bool>(config, "saveDebug", false);
	const bool saveColormapped = getOptional<bool>(config, "saveColormapped", true);
	const std::string colormap = getOptional<std::string>(config, "colormap", "");
	float colormapMinVal = getOptional<float>(config, "colormapMinVal", 0.0);
	float colormapMaxVal = getOptional<float>(config, "colormapMaxVal", 1.0);

	std::shared_ptr<Image<3>> solution = std::make_shared<Image<3>>(gridRes, gridRes);
	std::shared_ptr<Image<3>> dirichletdistance = std::make_shared<Image<3>>(gridRes, gridRes);
	std::shared_ptr<Image<3>> neumanndistance = std::make_shared<Image<3>>(gridRes, gridRes);
	std::shared_ptr<Image<3>> indomain = std::make_shared<Image<3>>(gridRes, gridRes);

	const std::string txtdir = getRequired<std::string>(config, "txtdir");
	std::ofstream valuefile(txtdir + "values.txt");
	std::ofstream samplesfile(txtdir + "samples.txt");

	for (int i = 0; i < gridRes; i++) {
		for (int j = 0; j < gridRes; j++) {
			int idx = i * gridRes + j;

			// debug / scene data
			float inDomain  = queries.insideDomain(samplePts[idx].pt) ? 1 : 0;
			float dirichletDist = samplePts[idx].dirichletDist;
			float neumannDist = samplePts[idx].neumannDist;
			dirichletdistance->get(j, i) = Array3(dirichletDist, 0.0, 0.0);
			neumanndistance->get(j, i) = Array3(0.0, neumannDist, 0.0);
			indomain->get(j, i) = Array3(0.0, 0.0, inDomain);

			// float dirichletVal = pde.dirichlet(samplePts[idx].pt);
			// float neumannVal = pde.neumann(samplePts[idx].pt);
			// float sourceVal = pde.source(samplePts[idx].pt);
			// // boundaryData->get(j, i) = Array3(dirichletVal, neumannVal, sourceVal);
			// boundaryData->get(j, i) = Array3(dirichletDist, neumannDist, 0.0);

			// solution data
			std::vector<float> zero{0.0f, 0.0f};
			const float *value = samplePts[idx].statistics ? samplePts[idx].statistics->getEstimatedGradient() : &zero[0];
			bool maskOutValue = (!inDomain && !isDoubleSided) ||
								std::min(std::abs(dirichletDist), std::abs(neumannDist)) < boundaryDistanceMask;
			
			solution->get(j, i) = Array3(maskOutValue ? 0.0f : value[1]);
			// float value = samplePts[idx].statistics ? samplePts[idx].statistics->getEstimatedSolution(): 0.0f;
			// bool maskOutValue = (!inDomain && !isDoubleSided) ||
			// 					std::min(std::abs(dirichletDist), std::abs(neumannDist)) < boundaryDistanceMask;
			// solution->get(j, i) = Array3(maskOutValue ? 0.0f : value);
			
			// valuefile<<(maskOutValue ? 0.0f : value)<<"\n";
			// samplesfile<<samplePts[idx].pt[0]<<", "<<samplePts[idx].pt[1]<<"\n";

			// if (value > colormapMaxVal) {
			// 	colormapMaxVal = value;
			// }
			// if (value < colormapMinVal) {
			// 	colormapMinVal = value;
			// }
		}
		// valuefile<<std::endl;
		// samplesfile<<std::endl;
	}
	// valuefile<<std::endl;
	// samplesfile<<std::endl;
	
	valuefile.close();
	samplesfile.close();

	std::cout<<"max_val: "<<colormapMaxVal<<", min_val: "<<colormapMinVal<<std::endl;

	// writeSolution(solutionFile, solution, boundaryDistance, boundaryData, saveDebug,
	// 			  saveColormapped, colormap, colormapMinVal, colormapMaxVal);
	writeSolution(solutionFile, solution, dirichletdistance, neumanndistance, indomain, saveDebug,
				  saveColormapped, colormap, colormapMinVal, colormapMaxVal);
}

void createEvaluationGrid(std::vector<zombie::EvaluationPoint<float, 2>>& evalPts,
						  const zombie::GeometricQueries<2> &queries,
						  const Vector2 &bMin, const Vector2 &bMax,
						  const int gridRes) {
	Vector2 extent = bMax - bMin;
	for (int i = 0; i < gridRes; i++) {
		for (int j = 0; j < gridRes; j++) {
			Vector2 pt((i / float(gridRes)) * extent.x() + bMin.x(),
					   (j / float(gridRes)) * extent.y() + bMin.y());
			float dDist = queries.computeDistToDirichlet(pt, false);
			float nDist = queries.computeDistToNeumann(pt, false);
			evalPts.emplace_back(zombie::EvaluationPoint<float, 2>(pt, Vector2::Zero(),
																   zombie::SampleType::InDomain,
																   dDist, nDist, 0.0f));
		}
	}
}

void saveEvaluationGrid(const std::vector<zombie::EvaluationPoint<float, 2>>& evalPts,
						const zombie::PDE<float, 2>& pde,
						const zombie::GeometricQueries<2> &queries,
						const bool isDoubleSided, const json &config) {
	const std::string solutionFile = getOptional<std::string>(config, "solutionFile", "solution.pfm");
	const int gridRes = getRequired<int>(config, "gridRes");
	const float boundaryDistanceMask = getOptional<float>(config, "boundaryDistanceMask", 0.0);

	const bool saveDebug = getOptional<bool>(config, "saveDebug", false);
	const bool saveColormapped = getOptional<bool>(config, "saveColormapped", true);
	const std::string colormap = getOptional<std::string>(config, "colormap", "");
	const float colormapMinVal = getOptional<float>(config, "colormapMinVal", 0.0);
	const float colormapMaxVal = getOptional<float>(config, "colormapMaxVal", 1.0);

	std::shared_ptr<Image<3>> solution = std::make_shared<Image<3>>(gridRes, gridRes);
	std::shared_ptr<Image<3>> dirichletdistance = std::make_shared<Image<3>>(gridRes, gridRes);
	std::shared_ptr<Image<3>> neumanndistance = std::make_shared<Image<3>>(gridRes, gridRes);
	std::shared_ptr<Image<3>> indomain = std::make_shared<Image<3>>(gridRes, gridRes);
	for (int i = 0; i < gridRes; i++) {
		for (int j = 0; j < gridRes; j++) {
			int idx = i * gridRes + j;

			// debug / scene data
			float inDomain  = queries.insideDomain(evalPts[idx].pt) ? 1 : 0;
			float dirichletDist = evalPts[idx].dirichletDist;
			float neumannDist = evalPts[idx].neumannDist;
			dirichletdistance->get(j, i) = Array3(dirichletDist, 0.0, 0.0);
			neumanndistance->get(j, i) = Array3(0.0, neumannDist, 0.0);
			indomain->get(j, i) = Array3(0.0, 0.0, inDomain);

			// float dirichletVal = pde.dirichlet(evalPts[idx].pt);
			// float neumannVal = pde.neumann(evalPts[idx].pt);
			// float sourceVal = pde.source(evalPts[idx].pt);
			// boundaryData->get(j, i) = Array3(dirichletVal, neumannVal, sourceVal);

			// solution data
			float value = evalPts[idx].getEstimatedSolution();
			bool maskOutValue = (!inDomain && !isDoubleSided) ||
								std::min(std::abs(dirichletDist), std::abs(neumannDist)) < boundaryDistanceMask;
			solution->get(j, i) = Array3(maskOutValue ? 0.0f : value);
		}
	}

	writeSolution(solutionFile, solution, dirichletdistance, neumanndistance, indomain, saveDebug,
				  saveColormapped, colormap, colormapMinVal, colormapMaxVal);
}
