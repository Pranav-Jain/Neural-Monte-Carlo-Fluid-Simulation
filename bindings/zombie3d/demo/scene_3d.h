#pragma once

#include <zombie/core/pde.h>
#include <zombie/utils/fcpw_scene_loader.h>
#include <fstream>
#include <sstream>
#include <algorithm>
#include "config.h"

class Scene {
public:
	fcpw::BoundingBox<3> bbox;
	std::vector<Vector3> vertices;
	std::vector<std::vector<size_t>> segments;

	const bool isWatertight;
	const bool isDoubleSided;

	zombie::GeometricQueries<3> queries;
	zombie::PDE<float, 3> pde;

	Scene(const json &config, std::vector<std::vector<std::vector<float>>> sourceValue_mat):
		  isWatertight(getOptional<bool>(config, "isWatertight", false)),
		  isDoubleSided(getOptional<bool>(config, "isDoubleSided", false)),
		  queries(isWatertight)
	{
		const std::string boundaryFile = getRequired<std::string>(config, "boundary");
		// const std::string isNeumannFile = getRequired<std::string>(config, "isNeumann");
		bool normalize = getOptional<bool>(config, "normalizeDomain", false);
		bool flipOrientation = getOptional<bool>(config, "flipOrientation", false);
		absorptionCoeff = getOptional<float>(config, "absorptionCoeff", 0.0f);

		sourceValue = sourceValue_mat;

		zombie::loadSurfaceMesh<3>(boundaryFile, vertices, segments);
        bbox = zombie::computeBoundingBox(vertices, false, 1.0);
		separateBoundaries();
		populateGeometricQueries();
		setPDE();
	}

	bool ignoreCandidateSilhouette(float dihedralAngle, int index) const {
		// ignore convex vertices/edges for closest silhouette point tests when solving an interior problem;
		// NOTE: for complex scenes with both open and closed meshes, the primitive index argument
		// (of an adjacent line segment/triangle in the scene) can be used to determine whether a
		// vertex/edge should be ignored as a candidate for silhouette tests.
		return isDoubleSided ? false : dihedralAngle < 1e-3f;
	}

private:
	void separateBoundaries() {
		std::function<bool(float, int)> ignoreCandidateSilhouette = [this](float dihedralAngle, int index) -> bool {
			return this->ignoreCandidateSilhouette(dihedralAngle, index);
		};
		dirichletSceneLoader = new zombie::FcpwSceneLoader<3>(dirichletVertices, dirichletSegments);
		neumannSceneLoader = new zombie::FcpwSceneLoader<3>(vertices, segments,
															ignoreCandidateSilhouette, true);
	}

	void populateGeometricQueries() {
		neumannSamplingTraversalWeight = [this](float r2) -> float {
			float r = std::max(std::sqrt(r2), 1e-2f);
			return std::fabs(this->harmonicGreensFn.evaluate(r));
		};

		const fcpw::Aggregate<3> *dirichletAggregate = dirichletSceneLoader->getSceneAggregate();
		const fcpw::Aggregate<3> *neumannAggregate = neumannSceneLoader->getSceneAggregate();
		zombie::populateGeometricQueries<3>(queries, bbox, dirichletAggregate, neumannAggregate,
											neumannSamplingTraversalWeight);
	}

	// void setPDE() {
	// 	// float maxLength = this->bbox.extent().maxCoeff();
	// 	Vector2 extent = this->bbox.extent();
	// 	pde.dirichlet = [this, extent](const Vector2& x) -> float {
	// 		Vector2 uv = (x - this->bbox.pMin).array() / extent.array();
	// 		return this->dirichletBoundaryValue->get(uv)[0];
	// 	};
	// 	pde.neumann = [this, extent](const Vector2& x) -> float {
	// 		// Vector2 uv = (x - this->bbox.pMin) / maxLength;
	// 		Vector2 uv = (x - this->bbox.pMin).array() / extent.array();
	// 		return this->neumannBoundaryValue->get(uv)[0];
	// 	};
	// 	pde.dirichletDoubleSided = [this, extent](const Vector2& x, bool _) -> float {
	// 		// Vector2 uv = (x - this->bbox.pMin) / maxLength;
	// 		Vector2 uv = (x - this->bbox.pMin).array() / extent.array();
	// 		return this->dirichletBoundaryValue->get(uv)[0];
	// 	};
	// 	pde.neumannDoubleSided = [this, extent](const Vector2& x, bool _) -> float {
	// 		// Vector2 uv = (x - this->bbox.pMin) / maxLength;
	// 		Vector2 uv = (x - this->bbox.pMin).array() / extent.array();
	// 		return this->neumannBoundaryValue->get(uv)[0];
	// 	};
	// 	pde.source = [this, extent](const Vector2& x) -> float {
	// 		// Vector2 uv = (x - this->bbox.pMin) / maxLength;
	// 		Vector2 uv = (x - this->bbox.pMin).array() / extent.array();
	// 		return this->sourceValue->get(uv)[0];
	// 	};
	// 	pde.absorption = absorptionCoeff;
	// }

	void setPDE() {
		Vector3 extent = this->bbox.extent();
		pde.dirichlet = [this, extent](const Vector3& x) -> float {
			// Vector3 uv = (x - this->bbox.pMin) / maxLength;
			return 0.0;
		};
		pde.neumann = [this, extent](const Vector3& x) -> float {
			// Vector3 uv = (x - this->bbox.pMin) / maxLength;
			return 0.0;
		};
		pde.dirichletDoubleSided = [this, extent](const Vector3& x, bool _) -> float {
			// Vector2 uv = (x - this->bbox.pMin) / maxLength;
			return 0.0;
		};
		pde.neumannDoubleSided = [this, extent](const Vector3& x, bool _) -> float {
			// Vector2 uv = (x - this->bbox.pMin) / maxLength;
			return 0.0;
		};
		pde.source = [this, extent](const Vector3& x) -> float {
			Vector3 uv = (x - this->bbox.pMin).array() / extent.array();
			int i = std::clamp(int(uv.x() * sourceValue.size()), 0, int(sourceValue.size() - 1));
			int j = std::clamp(int(uv.y() * sourceValue[0].size()), 0, int(sourceValue[0].size() - 1));
			int k = std::clamp(int(uv.z() * sourceValue[0][0].size()), 0, int(sourceValue[0][0].size() - 1));
			return sourceValue[i][j][k];
		};
		pde.absorption = absorptionCoeff;
	}


	std::vector<Vector3> dirichletVertices;
	std::vector<Vector3> neumannVertices;

	std::vector<std::vector<size_t>> dirichletSegments;
	std::vector<std::vector<size_t>> neumannSegments;

	zombie::FcpwSceneLoader<3>* dirichletSceneLoader;
	zombie::FcpwSceneLoader<3>* neumannSceneLoader;

    std::vector<std::vector<std::vector<float>>> sourceValue;
	float absorptionCoeff;

	zombie::HarmonicGreensFnFreeSpace<3> harmonicGreensFn;
	std::function<float(float)> neumannSamplingTraversalWeight;
};
