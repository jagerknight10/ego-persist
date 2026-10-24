# Geometry Agent Brief

## Mission

Project observations into the world frame and maintain track state.

## Owned Paths

- `src/egopersist/geometry`
- `src/egopersist/association`
- `src/egopersist/mapping`
- `tests/test_geometry*`
- `tests/test_association*`
- `tests/test_mapping*`

## Tasks

1. Implement pose and point transforms.
2. Implement depth back-projection.
3. Implement mask projection and visibility checks.
4. Implement 3D IoU and centroid distance.
5. Implement gated Hungarian association.
6. Implement provisional tracks.
7. Implement the map state machine.
8. Implement commit, defer, reject, and revert.

## Rules

- Use right-handed world coordinates.
- Use metres.
- Use radians internally.
- Reject invalid depth and calibration.
- Do not remove a map element on non-observation.
- Do not use future frames.

## Tests

- transform round-trip;
- projection round-trip;
- invalid depth handling;
- association gate;
- commit gate;
- occlusion defer;
- reversion gate.
