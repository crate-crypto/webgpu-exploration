#![allow(non_snake_case)]
use cudatowgsl::*;

#[test]
fn zero_points_test() {
	let gpu = pollster::block_on(GpuConsts::initialaze()).unwrap();

  let mut msm: MSMContext = MSMContext::new(100000, 100000);

	let mut projectiveResultsPtr: [u32; 96] = [0; 96];
	let mut scalarsPtr = [0; 32];

	let pointsFilePath = "./tests/Test_data/Zero_points.hex";
  let scalarsFilePath = "./tests/Test_data/Zero_scalars.hex";

	let points = u32_as_mut_slice_u8(&mut projectiveResultsPtr);
	MSMReadHexPoints(points, 1, pointsFilePath.into());

	let scalars = u32_as_mut_slice_u8(&mut scalarsPtr);
	MSMReadHexScalars(scalars, 1, scalarsFilePath.into());

	let affinePointsPtr: [u32; 24] = [0, 1,2,3,4,5,6,70,8,9,10,11, 0, 1,2,3,4,5,6,70,8,9,10,11]; 
	msm.msmPreprocessPoints(&affinePointsPtr, 65536, &gpu);

	let points = u8_as_mut_slice_u32(points);
	let points = u32_as_mut_slice_u64(points);
	let scalars = u8_as_slice_u32(scalars);

	msm.msmRun(points, &scalars, 65536, &gpu);

	let expected = vec![0, 1, 2231943168, 805306368, 386620740, 3121170432, 519266863, 16061327, 438491635, 1822509371, 3325756864, 398790890, 28195398, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
	assert_eq!(expected, msm.ml.results);
}

#[test]
fn max_points_test() {
	let gpu = pollster::block_on(GpuConsts::initialaze()).unwrap();

  let mut msm: MSMContext = MSMContext::new(100000, 100000);

	let mut projectiveResultsPtr: [u32; 96] = [0; 96];
	let mut scalarsPtr = [0; 32];

	let pointsFilePath = "./tests/Test_data/Max_points.hex";
  let scalarsFilePath = "./tests/Test_data/Max_scalars.hex";

	let points = u32_as_mut_slice_u8(&mut projectiveResultsPtr);
	MSMReadHexPoints(points, 1, pointsFilePath.into());

	let scalars = u32_as_mut_slice_u8(&mut scalarsPtr);
	MSMReadHexScalars(scalars, 1, scalarsFilePath.into());

	let affinePointsPtr: [u32; 24] = [0, 1,2,3,4,5,6,70,8,9,10,11, 0, 1,2,3,4,5,6,70,8,9,10,11]; 
	msm.msmPreprocessPoints(&affinePointsPtr, 65536, &gpu);

	let points = u8_as_mut_slice_u32(points);
	let points = u32_as_mut_slice_u64(points);
	let scalars = u8_as_slice_u32(scalars);

	msm.msmRun(points, &scalars, 65536, &gpu);

	let expected = vec![0, 1, 2231943168, 805306368, 386620740, 3121170432, 519266863, 16061327, 438491635, 1822509371, 3325756864, 398790890, 28195398, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

	assert_eq!(expected, msm.ml.results);
}

#[test]
fn normal_points_test() {
	let gpu = pollster::block_on(GpuConsts::initialaze()).unwrap();

  let mut msm: MSMContext = MSMContext::new(100000, 100000);

	let mut projectiveResultsPtr: [u32; 96] = [0; 96];
	let mut scalarsPtr = [0; 32];

	let pointsFilePath = "./Data/points.hex";
  let scalarsFilePath = "./Data/scalars.hex";

	let points = u32_as_mut_slice_u8(&mut projectiveResultsPtr);
	MSMReadHexPoints(points, 1, pointsFilePath.into());

	let scalars = u32_as_mut_slice_u8(&mut scalarsPtr);
	MSMReadHexScalars(scalars, 1, scalarsFilePath.into());

	let affinePointsPtr: [u32; 24] = [0, 1,2,3,4,5,6,70,8,9,10,11, 0, 1,2,3,4,5,6,70,8,9,10,11]; 
	msm.msmPreprocessPoints(&affinePointsPtr, 65536, &gpu);

	let points = u8_as_mut_slice_u32(points);
	let points = u32_as_mut_slice_u64(points);
	let scalars = u8_as_slice_u32(scalars);

	msm.msmRun(points, &scalars, 65536, &gpu);

	let expected = vec![1715505673, 3946063407, 3923780668, 2493396589, 2623681009, 2657285940, 3983840219, 2211592574, 3918225151, 3959960043, 3986814013, 2344699398, 4036056129, 3731235363, 3923958903, 3583171428, 2351260709, 3821750944, 3066524883, 4292347058, 3461982666, 2360206048, 2372206496, 1653095610];
	assert_eq!(expected, msm.gpuPointsMemory);
}