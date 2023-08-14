#![allow(non_snake_case)]
use cudatowgsl::*;

fn main() {
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


	let args: Vec<String> = std::env::args().collect();

	let baches = if args.len() > 1 {
			match args[1].parse::<u32>() {
					Ok(v) => v,
					Err(_) => {
							println!("Invalid command line argument. Used the default value of 1000");
							1000
					}
			}
	} else {
			1000
	};

	println!("Batch number: {}", baches);

	msm.msmRun(points, &scalars, 65536*baches, &gpu);

	// let cpu_points = vec![0.000014457, 0.00015908, 0.0015473, 0.016325, 0.16090, 1.5984, 15.929];
	// let gpu_points = vec![0.0079639, 0.0083614, 0.0094583, 0.018779, 0.10669, 0.96376, 1.4567];

	//build_plot(&cpu_points, &gpu_points).unwrap();
}
