#pragma once

#include <fcpw/core/primitive.h>

namespace fcpw {

template<size_t DIM, typename PrimitiveType=Primitive<DIM>, typename SilhouetteType=SilhouettePrimitive<DIM>>
class Baseline: public Aggregate<DIM> {
public:
	// constructor
	Baseline(const std::vector<PrimitiveType *>& primitives_,
			 const std::vector<SilhouetteType *>& silhouettes_);

	// returns bounding box
	BoundingBox<DIM> boundingBox() const;

	// returns centroid
	Vector<DIM> centroid() const;

	// returns surface area
	float surfaceArea() const;

	// returns signed volume
	float signedVolume() const;

	// intersects with ray, starting the traversal at the specified node in an aggregate
	// NOTE: interactions are invalid when checkForOcclusion is enabled
	int intersectFromNode(Ray<DIM>& r, std::vector<Interaction<DIM>>& is,
						  int nodeStartIndex, int aggregateIndex, int& nodesVisited,
						  bool checkForOcclusion=false, bool recordAllHits=false) const;

	// intersects with sphere, starting the traversal at the specified node in an aggregate
	// NOTE: interactions contain primitive index
	int intersectFromNode(const BoundingSphere<DIM>& s,
						  std::vector<Interaction<DIM>>& is,
						  int nodeStartIndex, int aggregateIndex,
						  int& nodesVisited, bool recordOneHit=false,
						  const std::function<float(float)>& primitiveWeight={}) const;

	// intersects with sphere, starting the traversal at the specified node in an aggregate
	// NOTE: interactions contain primitive index
	int intersectStochasticFromNode(const BoundingSphere<DIM>& s,
									std::vector<Interaction<DIM>>& is, float *randNums,
									int nodeStartIndex, int aggregateIndex, int& nodesVisited,
									const std::function<float(float)>& traversalWeight={},
									const std::function<float(float)>& primitiveWeight={}) const;

	// finds closest point to sphere center, starting the traversal at the specified node in an aggregate
	bool findClosestPointFromNode(BoundingSphere<DIM>& s, Interaction<DIM>& i,
								  int nodeStartIndex, int aggregateIndex,
								  int& nodesVisited, bool recordNormal=false) const;

	// finds closest silhouette point to sphere center, starting the traversal at the specified node in an aggregate
	bool findClosestSilhouettePointFromNode(BoundingSphere<DIM>& s, Interaction<DIM>& i,
									   		int nodeStartIndex, int aggregateIndex,
									   		int& nodesVisited, bool flipNormalOrientation=false,
											float squaredMinRadius=0.0f, float precision=1e-3f,
											bool recordNormal=false) const;

protected:
	// members
	const std::vector<PrimitiveType *>& primitives;
	const std::vector<SilhouetteType *>& silhouettes;
	bool primitiveTypeIsAggregate;
};

} // namespace fcpw

#include "baseline.inl"
