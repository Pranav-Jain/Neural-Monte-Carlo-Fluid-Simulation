#pragma once

#include <zombie/core/pde.h>
#include <zombie/utils/fcpw_scene_loader.h>
#include <fstream>
#include <sstream>
#include "config.h"
#include "image.h"

class Scene {
public:
	fcpw::BoundingBox<2> bbox;
	std::vector<Vector2> vertices;
	std::vector<std::vector<size_t>> segments;

	const bool isWatertight;
	const bool isDoubleSided;

	zombie::GeometricQueries<2> queries;
	zombie::PDE<float, 2> pde;

	Scene(const json &config):
		  isWatertight(getOptional<bool>(config, "isWatertight", true)),
		  isDoubleSided(getOptional<bool>(config, "isDoubleSided", false)),
		  queries(isWatertight)
	{
		const std::string boundaryFile = getRequired<std::string>(config, "boundary");
		// const std::string isNeumannFile = getRequired<std::string>(config, "isNeumann");
		// const std::string dirichletBoundaryValueFile = getRequired<std::string>(config, "dirichletBoundaryValue");
		// const std::string neumannBoundaryValueFile = getRequired<std::string>(config, "neumannBoundaryValue");
		const std::string sourceValueFile = getRequired<std::string>(config, "sourceValue");
		bool normalize = getOptional<bool>(config, "normalizeDomain", false);
		bool flipOrientation = getOptional<bool>(config, "flipOrientation", true);
		absorptionCoeff = getOptional<float>(config, "absorptionCoeff", 0.0f);

		sourceValue = std::make_shared<Image<1>>(sourceValueFile);
		int h = sourceValue->h;
		int w = sourceValue->w;
		isNeumann = std::make_shared<Image<1>>(h, w, 1.0);
		dirichletBoundaryValue = std::make_shared<Image<1>>(h, w, 0.0);
		neumannBoundaryValue = std::make_shared<Image<1>>(h, w, 0.0);

		// isNeumann = std::make_shared<Image<1>>(isNeumannFile);
		// dirichletBoundaryValue = std::make_shared<Image<1>>(dirichletBoundaryValueFile);
		// neumannBoundaryValue = std::make_shared<Image<1>>(neumannBoundaryValueFile);
		

		loadOBJ(boundaryFile, normalize, flipOrientation);
		separateBoundaries();
		populateGeometricQueries();
		setPDE();
	}

	Scene(const json &config, std::vector<std::vector<float>> sourceValue_mat):
		  isWatertight(getOptional<bool>(config, "isWatertight", false)),
		  isDoubleSided(getOptional<bool>(config, "isDoubleSided", false)),
		  queries(isWatertight)
	{
		const std::string boundaryFile = getRequired<std::string>(config, "boundary");
		// const std::string isNeumannFile = getRequired<std::string>(config, "isNeumann");
		bool normalize = getOptional<bool>(config, "normalizeDomain", false);
		bool flipOrientation = getOptional<bool>(config, "flipOrientation", false);
		absorptionCoeff = getOptional<float>(config, "absorptionCoeff", 0.0f);

		int h = sourceValue_mat.size();
		int w = sourceValue_mat[0].size();
		// isNeumann = std::make_shared<Image<1>>(isNeumannFile);
		isNeumann = std::make_shared<Image<1>>(h, w, 1.0);
		dirichletBoundaryValue = std::make_shared<Image<1>>(h, w, 0.0);
		neumannBoundaryValue = std::make_shared<Image<1>>(h, w, 0.0);
		sourceValue = std::make_shared<Image<1>>(sourceValue_mat);

		loadOBJ(boundaryFile, normalize, flipOrientation);
		separateBoundaries();
		populateGeometricQueries();
		setPDE();
	}

	bool onNeumannBoundary(Vector2 x) const {
		Vector2 uv = (x - bbox.pMin) / bbox.extent().maxCoeff();
		return isNeumann->get(uv)[0] > 0;
	}

	bool ignoreCandidateSilhouette(float dihedralAngle, int index) const {
		// ignore convex vertices/edges for closest silhouette point tests when solving an interior problem;
		// NOTE: for complex scenes with both open and closed meshes, the primitive index argument
		// (of an adjacent line segment/triangle in the scene) can be used to determine whether a
		// vertex/edge should be ignored as a candidate for silhouette tests.
		return isDoubleSided ? false : dihedralAngle < 1e-3f;
	}

	float getSolveRegionVolume() const {
		if (isDoubleSided) return (bbox.pMax - bbox.pMin).prod();
		float solveRegionVolume = 0.0f;
		const fcpw::Aggregate<3> *dirichletAggregate = dirichletSceneLoader->getSceneAggregate();
		const fcpw::Aggregate<3> *neumannAggregate = neumannSceneLoader->getSceneAggregate();
		if (dirichletAggregate != nullptr) solveRegionVolume += dirichletAggregate->signedVolume();
		if (neumannAggregate != nullptr) solveRegionVolume += neumannAggregate->signedVolume();
		return std::fabs(solveRegionVolume);
	}


private:
	void loadOBJ(const std::string &filename, bool normalize, bool flipOrientation) {
		std::ifstream obj(filename);
		if (!obj) {
			std::cerr << "Error opening file: " << filename << std::endl;
			abort();
		}

		std::string line;
		while (std::getline(obj, line)) {
			std::istringstream ss(line);
			std::string token;
			ss >> token;
			if (token == "v") {
				float x, y;
				ss >> x >> y;
				vertices.emplace_back(Vector2(x, y));
			} else if (token == "l") {
				size_t i, j;
				ss >> i >> j;
				if (flipOrientation) {
					segments.emplace_back(std::vector<size_t>({j - 1, i - 1}));
				} else {
					segments.emplace_back(std::vector<size_t>({i - 1, j - 1}));
				}
			}
		}
		obj.close();

		if (normalize) {
			Vector2 cm(0, 0);
			for (Vector2 v : vertices) cm += v;
			cm /= vertices.size();
			float radius = 0.0f;
			for (Vector2& v : vertices) {
				v -= cm;
				radius = std::max(radius, v.norm());
			}
			for (Vector2& v : vertices) v /= radius;
		}

		bbox = zombie::computeBoundingBox(vertices, false, 1.0);
	}

	void separateBoundaries() {
		std::function<bool(float, int)> ignoreCandidateSilhouette = [this](float dihedralAngle, int index) -> bool {
			return this->ignoreCandidateSilhouette(dihedralAngle, index);
		};
		dirichletSceneLoader = new zombie::FcpwSceneLoader<2>(dirichletVertices, dirichletSegments);
		neumannSceneLoader = new zombie::FcpwSceneLoader<2>(vertices, segments,
															ignoreCandidateSilhouette, true);
	}

	void populateGeometricQueries() {
		neumannSamplingTraversalWeight = [this](float r2) -> float {
			float r = std::max(std::sqrt(r2), 1e-2f);
			return std::fabs(this->harmonicGreensFn.evaluate(r));
		};

		const fcpw::Aggregate<3> *dirichletAggregate = dirichletSceneLoader->getSceneAggregate();
		const fcpw::Aggregate<3> *neumannAggregate = neumannSceneLoader->getSceneAggregate();
		zombie::populateGeometricQueries<2>(queries, bbox, dirichletAggregate, neumannAggregate,
											neumannSamplingTraversalWeight);
	}

	void setPDE() {
		// float maxLength = this->bbox.extent().maxCoeff();
		Vector2 extent = this->bbox.extent();
		pde.dirichlet = [this, extent](const Vector2& x) -> float {
			// Vector2 uv = (x - this->bbox.pMin).array() / extent.array();
			// return this->dirichletBoundaryValue->get(uv)[0];
			return 0.0;
		};
		pde.neumann = [this, extent](const Vector2& x) -> float {
			// Vector2 uv = (x - this->bbox.pMin) / maxLength;
			// Vector2 uv = (x - this->bbox.pMin).array() / extent.array();
			// return this->neumannBoundaryValue->get(uv)[0];
			return 0.0;
		};
		pde.dirichletDoubleSided = [this, extent](const Vector2& x, bool _) -> float {
			// Vector2 uv = (x - this->bbox.pMin) / maxLength;
			// Vector2 uv = (x - this->bbox.pMin).array() / extent.array();
			// return this->dirichletBoundaryValue->get(uv)[0];
			return 0.0;
		};
		pde.neumannDoubleSided = [this, extent](const Vector2& x, bool _) -> float {
			// Vector2 uv = (x - this->bbox.pMin) / maxLength;
			// Vector2 uv = (x - this->bbox.pMin).array() / extent.array();
			// return this->neumannBoundaryValue->get(uv)[0];
			return 0.0;
		};
		pde.source = [this, extent](const Vector2& x) -> float {
			// Vector2 uv = (x - this->bbox.pMin) / maxLength;
			Vector2 uv = (x - this->bbox.pMin).array() / extent.array();
			return this->sourceValue->get(uv)[0];
		};
		pde.absorption = absorptionCoeff;
	}

	// void setPDE() {
	// 	float maxLength = this->bbox.extent().maxCoeff();
	// 	pde.dirichlet = [this, maxLength](const Vector2& x) -> float {
	// 		Vector2 uv = (x - this->bbox.pMin) / maxLength;
	// 		return this->dirichletBoundaryValue->get(uv)[0];
	// 	};
	// 	pde.neumann = [this, maxLength](const Vector2& x) -> float {
	// 		Vector2 uv = (x - this->bbox.pMin) / maxLength;
	// 		return this->neumannBoundaryValue->get(uv)[0];
	// 	};
	// 	pde.dirichletDoubleSided = [this, maxLength](const Vector2& x, bool _) -> float {
	// 		Vector2 uv = (x - this->bbox.pMin) / maxLength;
	// 		return this->dirichletBoundaryValue->get(uv)[0];
	// 	};
	// 	pde.neumannDoubleSided = [this, maxLength](const Vector2& x, bool _) -> float {
	// 		Vector2 uv = (x - this->bbox.pMin) / maxLength;
	// 		return this->neumannBoundaryValue->get(uv)[0];
	// 	};
	// 	pde.source = [this, maxLength](const Vector2& x) -> float {
	// 		Vector2 uv = (x - this->bbox.pMin) / maxLength;
	// 		return this->sourceValue->get(uv)[0];
	// 	};
	// 	pde.absorption = absorptionCoeff;
	// }


	std::vector<Vector2> dirichletVertices;
	std::vector<Vector2> neumannVertices;

	std::vector<std::vector<size_t>> dirichletSegments;
	std::vector<std::vector<size_t>> neumannSegments;

	zombie::FcpwSceneLoader<2>* dirichletSceneLoader;
	zombie::FcpwSceneLoader<2>* neumannSceneLoader;

	std::shared_ptr<Image<1>> isNeumann;
	std::shared_ptr<Image<1>> dirichletBoundaryValue;
	std::shared_ptr<Image<1>> neumannBoundaryValue;
	std::shared_ptr<Image<1>> sourceValue;
	float absorptionCoeff;

	zombie::HarmonicGreensFnFreeSpace<3> harmonicGreensFn;
	std::function<float(float)> neumannSamplingTraversalWeight;
};
