# Geometric Transformations
1. Rigid Transformations (Euclidean Transformations)
These are the simplest transformations that preserve the "rigidness" of an object.
Properties: Preserve distances, angles, and area. An object's shape and size remain identical.
What they include:
Translation (Pan): Moving an object without changing its orientation.
Rotation: Rotating an object around a point.
Example: Sliding a piece of paper across a desk or spinning it in place.
2. Similarity Transformations
This is a superset of rigid transformations.
Properties: Preserve angles and ratios of distances. Shapes are preserved, but the overall size can change.
What they include:
Rigid transformations (Translation, Rotation).
Uniform Scale: Scaling an object by the same factor in all directions (zooming in or out).
Example: Using a photocopier to enlarge or shrink a document. The shape is the same, but the size is different.
3. Affine Transformations
This is a superset of similarity transformations.
Properties: Preserve parallelism of lines. Distances and angles are not preserved. Straight lines remain straight.
What they include:
Similarity transformations (Translation, Rotation, Uniform Scale).
Non-uniform Scale: Scaling by different factors in the x and y directions (stretching or squashing).
Shear (Skew): Tilting one axis of an object, like pushing the top of a deck of cards to the side.
Example: The shadow of a rectangular window cast on a non-perpendicular wall.
4. Projective Transformations (Homographies)
This is a superset of affine transformations and is what your get_perspective_view_matrix function generates.
Properties: The most general linear transformation. It only preserves straight lines. Parallel lines may converge to a vanishing point.
What they include:
All Affine transformations.
Perspective Transform: Simulates the effect of viewing a 2D plane from a 3D viewpoint, like looking at a road stretching to the horizon.
Example: A photo of a rectangular door taken from an angle. The door appears as a trapezoid in the photo.
All the transformations in your code (identity, translate, scale, rotate, shear, perspective_view) fall into one of these four categories. They can all be represented by a 3x3 matrix and all map straight lines to straight lines.