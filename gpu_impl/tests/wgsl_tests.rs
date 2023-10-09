#![allow(non_snake_case)]
use cudatowgsl::*;

	// #[test]
	// fn check_time() {
	// 	let r = vec![1; 1000000];

  // 	let mut bindings: Bindings = Bindings::initialize_one(r);

	// 	let t1 = std::time::Instant::now();
	// 	let gpu = pollster::block_on(GpuConsts::initialaze()).unwrap();
	// 	let macro_time = std::time::Instant::now() - t1;

	// 	let t1 = std::time::Instant::now();
	// 	let bc = BufCoder::initialize(&gpu, &mut bindings, "add_two_vec_call", 1);
	// 	let buffer_time = std::time::Instant::now() - t1;

	// 	let t1 = std::time::Instant::now();
	// 	let _ = pollster::block_on(gpu.run(&bc)).unwrap();
	// 	let wgsl_time = std::time::Instant::now() - t1;

	// 	dbg!(macro_time);
	// 	dbg!(buffer_time);
	// 	dbg!(wgsl_time);

	// 	// let t1 = std::time::Instant::now();
	// 	// let res2 = pollster::block_on(run(&mut bindings, "qTerm_test", 1));
	// 	// let old_version_time = std::time::Instant::now() - t1;
	// 	// dbg!(old_version_time);

	// 	assert!(false);
	// }

	#[test]
	fn qTerm_test() {
		let input = vec![100];
		let expected = vec![4294967196];
		let mut bindings: Bindings = Bindings::initialize_one(input);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "qTerm_test", 1)));
	}

	#[test]
	fn initialize_PointXY_test() {
		let input = vec![0, 0, 0,0,0,0,0,0,0,0,0,0, 0, 0, 0,0,0,0,0,0,0,0,0,0];
		let expected = vec![0, 1,2,3,4,5,6,7,8,9,10,11,0, 1,2,3,4,5,6,7,8,9,10,11];
		let mut bindings: Bindings = Bindings::initialize_one(input);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "initialize_PointXY_test", 1)));
	}

	#[test]
	fn isEqual_test() {
		let input = vec![0];
		let expected = vec![1];
		let mut bindings: Bindings = Bindings::initialize_one(input);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "isEqual_test", 1)));
	}

	#[test]
	fn set_test() {
		let input = vec![0, 0, 0,0,0,0,0,0,0,0,0,0];
		let expected = vec![0, 1,2,3,4,5,6,7,8,9,10,11];
		let mut bindings: Bindings = Bindings::initialize_one(input);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "set_test", 1)));
	}

	#[test]
	fn swap_test() {
		let input = vec![0, 0, 0,0,0,0,0,0,0,0,0,0, 0, 0, 0,0,0,0,0,0,0,0,0,0];
		let expected = vec![0, 1,2,3,4,5,6,7,8,9,10,11,0, 0, 0,0,0,0,0,0,0,0,0,0];
		let mut bindings: Bindings = Bindings::initialize_one(input);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "swap_test", 1)));
	}

	#[test]
	fn initialize_PointXYZZ_test() {
		let input = vec![0, 0, 0,0,0,0,0,0,0,0,0,0, 0, 0, 0,0,0,0,0,0,0,0,0,0];
		let expected = vec![0, 1,2,3,4,5,6,7,8,9,10,11,0, 1,2,3,4,5,6,7,8,9,10,11];
		let mut bindings: Bindings = Bindings::initialize_one(input);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "initialize_PointXYZZ_test", 1)));
	}

	#[test]
	fn load_test() {
		let input = vec![0, 0,0,0,0,0,0,0,0,0,0,0];
		let expected = vec![0, 1,2,3,4,5,6,7,8,9,10,11];
		let mut bindings: Bindings = Bindings::initialize_one(input);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "load_test", 1)));
	}

	#[test]
	fn store_test() {
		let input = vec![0, 0,0,0,0,0,0,0,0,0,0,0];
		let expected = vec![0, 1,2,3,4,5,6,7,8,9,10,11];
		let mut bindings: Bindings = Bindings::initialize_one(input);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "store_test", 1)));
	}

	#[test]
	fn load_PointXY_test() {
		let input = vec![0, 0,0,0,0,0,0,0,0,0,0,0, 0, 0,0,0,0,0,0,0,0,0,0,0];
		let expected = vec![0, 1,2,3,4,5,6,7,8,9,10,11,0, 1,2,3,4,5,6,7,8,9,10,100];
		let mut bindings: Bindings = Bindings::initialize_one(input);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "load_PointXY_test", 1)));
	}

	#[test]
	fn loadUnaligned_PointXY_test() {
		let input = vec![0, 0,0,0,0,0,0,0,0,0,0,0, 0, 0,0,0,0,0,0,0,0,0,0,0];
		let expected = vec![0, 1,2,3,4,5,6,7,8,9,10,11,0, 1,2,3,4,5,6,7,8,9,10,101];
		let mut bindings: Bindings = Bindings::initialize_one(input);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "loadUnaligned_PointXY_test", 1)));
	}

	#[test]
	fn store_PointXY_test() {
		let input = vec![0, 0,0,0,0,0,0,0,0,0,0,0, 0, 0,0,0,0,0,0,0,0,0,0,0];
		let expected = vec![0, 1,2,3,4,5,6,7,8,9,10,11,0, 1,2,3,4,5,6,7,8,9,10,101];
		let mut bindings: Bindings = Bindings::initialize_one(input);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "store_PointXY_test", 1)));
	}

	#[test]
	fn load_PointXYZZ_test() {
		let input = vec![0, 0, 0,0,0,0,0,0,0,0,0,0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0, 0, 0,0,0,0,0,0,0,0,0,0,0, 0, 0,0,0,0,0,0,0,0,0,0,];
		let expected = vec![0, 1,2,3,4,5,6,7,8,9,10,11,0, 1,2,3,4,5,6,7,8,9,10,100,0, 1,2,3,4,5,6,7,8,9,10,11,0, 1,2,3,4,5,6,7,8,9,10,200];
		let mut bindings: Bindings = Bindings::initialize_one(input);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "load_PointXYZZ_test", 1)));
	}

	#[test]
	fn loadUnaligned_PointXYZZ_test() {
		let input = vec![0, 0, 0,0,0,0,0,0,0,0,0,0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0, 0, 0,0,0,0,0,0,0,0,0,0,0, 0, 0,0,0,0,0,0,0,0,0,0,];
		let expected = vec![0, 1,2,3,4,5,6,7,8,9,10,11,0, 1,2,3,4,5,6,7,8,9,10,100,0, 1,2,3,4,5,6,7,8,9,10,11,0, 1,2,3,4,5,6,7,8,9,10,200];
		let mut bindings: Bindings = Bindings::initialize_one(input);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "loadUnaligned_PointXYZZ_test", 1)));
	}

	#[test]
	fn store_PointXYZZ_test() {
		let input = vec![0, 0, 0,0,0,0,0,0,0,0,0,0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0, 0, 0,0,0,0,0,0,0,0,0,0,0, 0, 0,0,0,0,0,0,0,0,0,0,];
		let expected = vec![0, 1,2,3,4,5,6,7,8,9,10,11,0, 1,2,3,4,5,6,7,8,9,10,101,0, 1,2,3,4,5,6,7,8,9,10,11,0, 1,2,3,4,5,6,7,8,9,10,200];
		let mut bindings: Bindings = Bindings::initialize_one(input);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "store_PointXYZZ_test", 1)));
	}

	#[test]
	fn setZero_HighThroughput_test() {
		let input = vec![0, 0];
		let expected = vec![1, 0];
		let mut bindings: Bindings = Bindings::initialize_one(input);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "setZero_HighThroughput_test", 1)));
	}

	#[test]
	fn load_HighThroughput_test() {
		let input = vec![0, 0, 0,0,0,0,0,0,0,0,0,0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0, 0, 0,0,0,0,0,0,0,0,0,0,0, 0, 0,0,0,0,0,0,0,0,0,0,6,7];
		let expected = vec![0, 1,2,3,4,5,6,7,8,9,10,11,0, 1,2,3,4,5,6,7,8,9,10,100,0, 1,2,3,4,5,6,7,8,9,10,11,0, 1,2,3,4,5,6,7,8,9,10,200,0,0];
		let mut bindings: Bindings = Bindings::initialize_one(input);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "load_HighThroughput_test", 1)));
	}

	#[test]
	fn initialize_chain_t1_test() {
		let input = vec![0];
		let expected = vec![1];
		let mut bindings: Bindings = Bindings::initialize_one(input);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "initialize_chain_t1_test", 1)));
	}

	#[test]
	fn initialize_chain_t2_test() {
		let input = vec![0];
		let expected = vec![0];
		let mut bindings: Bindings = Bindings::initialize_one(input);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "initialize_chain_t2_test", 1)));
	}

	#[test]
	fn reset1_test() {
		let input = vec![0];
		let expected = vec![1];
		let mut bindings: Bindings = Bindings::initialize_one(input);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "reset1_test", 1)));
	}

	#[test]
	fn reset2_test() {
		let input = vec![0];
		let expected = vec![0];
		let mut bindings: Bindings = Bindings::initialize_one(input);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "reset2_test", 1)));
	}

	#[test]
	fn getCarry_test() {
		let input = vec![0];
		let expected = vec![0];
		let mut bindings: Bindings = Bindings::initialize_one(input);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "getCarry_test", 1)));
	}

	#[test]
	fn add_chain_t_test() {
		let input = vec![100, 200];
		let expected = vec![300, 0];
		let mut bindings: Bindings = Bindings::initialize_one(input);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "add_chain_t_test", 1)));
	}

	#[test]
	fn sub_chain_t_test() {
		let input = vec![500, 499];
		let expected = vec![1, 0];
		let mut bindings: Bindings = Bindings::initialize_one(input);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "sub_chain_t_test", 1)));
	}

	#[test]
	fn madwide1_test() {
		let input = vec![1, 2, 200, 100];
		let expected = vec![201, 102, 0, 100];
		let mut bindings: Bindings = Bindings::initialize_one(input);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "madwide1_test", 1)));
	}

	#[test]
	fn computeNP0_test() {
		let input = vec![1];
		let expected = vec![4294967295];
		let mut bindings: Bindings = Bindings::initialize_one(input);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "computeNP0_test", 1)));
	}

	#[test]
	fn mp_zero_test() {
		let input = vec![0, 1,2,3,4,5,6,7,8,9,10,11];
		let expected = vec![0, 0,0,0,0,0,0,0,0,0,0,0];
		let mut bindings: Bindings = Bindings::initialize_one(input);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "mp_zero_test", 1)));
	}

	#[test]
	fn mp_copy_test() {
		let input = vec![0, 0,0,0,0,0,0,0,0,0,0,0];
		let expected = vec![0, 1,2,3,4,5,6,7,8,9,10,11];
		let mut bindings: Bindings = Bindings::initialize_one(input);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "mp_copy_test", 1)));
	}

	#[test]
	fn mp_logical_or_test() {
		let input = vec![0, 1,2,3,4,5,6,7,8,9,10,11];
		let expected = vec![15, 1,2,3,4,5,6,7,8,9,10,11];
		let mut bindings: Bindings = Bindings::initialize_one(input);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "mp_logical_or_test", 1)));
	}

	#[test]
	fn mp_shift_right_test() {
		let input = vec![0, 1,2,3,4,5,6,7,8,9,10,11];
		let expected = vec![0, 1, 1, 3, 2, 3, 3, 7, 4, 5, 5, 5];
		let mut bindings: Bindings = Bindings::initialize_one(input);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "mp_shift_right_test", 1)));
	}

	#[test]
	fn mp_shift_left_test() {
		let input = vec![0, 1,2,3,4,5,6,7,8,9,10,11];
		let expected = vec![0, 2, 6, 6, 14, 10, 14, 14, 30, 18, 22, 22];
		let mut bindings: Bindings = Bindings::initialize_one(input);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "mp_shift_left_test", 1)));
	}

	#[test]
	fn mp_add_test() {
		let input = vec![0, 0,0,0,0,0,0,0,0,0,0,0];
		let expected = vec![0, 101, 202, 303, 404, 505, 606, 707, 808, 909, 1010, 1111];
		let mut bindings: Bindings = Bindings::initialize_one(input);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "mp_add_test", 1)));
	}

	#[test]
	fn mp_add_carry_test() {
		let input = vec![0, 0,0,0,0,0,0,0,0,0,0,0];
		let expected = vec![0, 101, 202, 303, 404, 505, 606, 707, 808, 909, 1010, 1111];
		let mut bindings: Bindings = Bindings::initialize_one(input);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "mp_add_carry_test", 1)));
	}

	#[test]
	fn mp_sub_test() {
		let input = vec![0, 0,0,0,0,0,0,0,0,0,0,0];
		let expected = vec![0, 99, 198, 297, 396, 495, 594, 693, 792, 891, 990, 1089];
		let mut bindings: Bindings = Bindings::initialize_one(input);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "mp_sub_test", 1)));
	}

	#[test]
	fn mp_sub_carry_test() {
		let input = vec![0, 0,0,0,0,0,0,0,0,0,0,0];
		let expected = vec![0, 99, 198, 297, 396, 495, 594, 693, 792, 891, 990, 1089];
		let mut bindings: Bindings = Bindings::initialize_one(input);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "mp_sub_carry_test", 1)));
	}

	#[test]
	fn mp_comp_eq_test() {
		let input = vec![100];
		let expected = vec![1];
		let mut bindings: Bindings = Bindings::initialize_one(input);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "mp_comp_eq_test", 1)));
	}

	#[test]
	fn mp_comp_ge_test() {
		let input = vec![100];
		let expected = vec![0];
		let mut bindings: Bindings = Bindings::initialize_one(input);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "mp_comp_ge_test", 1)));
	}

	#[test]
	fn mp_comp_gt_test() {
		let input = vec![100];
		let expected = vec![1];
		let mut bindings: Bindings = Bindings::initialize_one(input);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "mp_comp_gt_test", 1)));
	}

	#[test]
	fn mp_select_test() {
		let input = vec![0, 0,0,0,0,0,0,0,0,0,0,0];
		let expected = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
		let mut bindings: Bindings = Bindings::initialize_one(input);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "mp_select_test", 1)));
	}

	#[test]
	fn mp_mul_red_cl_test() {
		let input = vec![100];
		let expected = vec![0];
		let mut bindings: Bindings = Bindings::initialize_one(input);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "mp_mul_red_cl_test", 1)));
	}

	#[test]
	fn mp_sqr_red_cl_test() {
		let input = vec![0, 0,0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0];
		let expected = vec![0, 0,0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0];
		let mut bindings: Bindings = Bindings::initialize_one(input);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "mp_sqr_red_cl_test", 1)));
	}

	#[test]
	fn add_test() {
		let input = vec![0, 0,0,0,0,0,0,0,0,0,0,0];
		let expected = vec![0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 112];
		let mut bindings: Bindings = Bindings::initialize_one(input);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "add_test", 1)));
	}

	#[test]
	fn sub_test() {
		let input = vec![0, 0,0,0,0,0,0,0,0,0,0,0];
		let expected = vec![0, 9, 18, 27, 36, 45, 54, 63, 72, 81, 90, 90];
		let mut bindings: Bindings = Bindings::initialize_one(input);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "sub_test", 1)));
	}

	#[test]
	fn addN_test() {
		let input = vec![0, 0,0,0,0,0,0,0,0,0,0,0];
		let expected = vec![1, 2231943169, 805306370, 386620743, 3121170436, 519266868, 16061333, 438491642, 1822509379, 3325756873, 398790900, 28195499];
		let mut bindings: Bindings = Bindings::initialize_one(input);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "addN_test", 1)));
	}

	#[test]
	fn add2N_test() {
		let input = vec![0, 0,0,0,0,0,0,0,0,0,0,0];
		let expected = vec![2, 168919041, 1610612739, 773241483, 1947373572, 1038533732, 32122660, 876983277, 3645018750, 2356546441, 797581791, 56390897];
		let mut bindings: Bindings = Bindings::initialize_one(input);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "add2N_test", 1)));
	}

	#[test]
	fn add3N_test() {
		let input = vec![0, 0,0,0,0,0,0,0,0,0,0,0];
		let expected = vec![3, 2400862209, 2415919107, 1159862223, 773576708, 1557800596, 48183987, 1315474912, 1172560825, 1387336010, 1196372682, 84586295];
		let mut bindings: Bindings = Bindings::initialize_one(input);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "add3N_test", 1)));
	}

	#[test]
	fn add4N_test() {
		let input = vec![0, 0,0,0,0,0,0,0,0,0,0,0];
		let expected = vec![4, 337838081, 3221225476, 1546482963, 3894747140, 2077067459, 64245314, 1753966547, 2995070196, 418125578, 1595163573, 112781693];
		let mut bindings: Bindings = Bindings::initialize_one(input);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "add4N_test", 1)));
	}

	#[test]
	fn add5N_test() {
		let input = vec![0, 0,0,0,0,0,0,0,0,0,0,0];
		let expected = vec![5, 2569781249, 4026531844, 1933103703, 2720950276, 2596334323, 80306641, 2192458182, 522612271, 3743882443, 1993954463, 140977091];
		let mut bindings: Bindings = Bindings::initialize_one(input);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "add5N_test", 1)));
	}

	#[test]
	fn add6N_test() {
		let input = vec![0, 0,0,0,0,0,0,0,0,0,0,0];
		let expected = vec![6, 506757121, 536870917, 2319724444, 1547153412, 3115601187, 96367968, 2630949817, 2345121642, 2774672011, 2392745354, 169172489];
		let mut bindings: Bindings = Bindings::initialize_one(input);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "add6N_test", 1)));
	}

	#[test]
	fn negateAddN_test() {
		let input = vec![10, 0,0,0,0,0,0,0,0,0,0,0];
		let expected = vec![4294967295, 2063024129, 3489660930, 3908346559, 1173796868, 3775700438, 4278905975, 3856475668, 2472457933, 969210441, 3896176416, 4266771999];
		let mut bindings: Bindings = Bindings::initialize_one(input);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "negateAddN_test", 1)));
	}

	#[test]
	fn negateAdd4N_test() {
		let input = vec![10, 0,0,0,0,0,0,0,0,0,0,0];
		let expected = vec![4294967292, 3957129217, 1073741824, 2748484339, 400220164, 2217899847, 4230721994, 2541000763, 1299897116, 3876841736, 2699803743, 4182185805];
		let mut bindings: Bindings = Bindings::initialize_one(input);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "negateAdd4N_test", 1)));
	}

	#[test]
	fn isZero_test() {
		let input = vec![0, 0,0,0,0,0,0,0,0,0,0,0];
		let memory = vec![0, 0,0,0,0,0,0,0,0,0,0,0];
		let expected = vec![1, 0,0,0,0,0,0,0,0,0,0,0];
		let mut bindings: Bindings = Bindings::initialize_two(input, memory);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "isZero_test", 2)));
	}

	#[test]
	fn mp_merge_cl_test() {
		let input = vec![0, 0,0,0,0,0,0,0,0,0,0,0];
		let expected = vec![100, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 100];
		let mut bindings: Bindings = Bindings::initialize_one(input);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "mp_merge_cl_test", 1)));
	}

	#[test]
	fn loadShared_test() {
		let input = vec![0, 0,0,0,0,0,0,0,0,0,0,0];
		let memory = vec![1, 2,0,4,6,8,2,3,4,5,2,4,4, 5,6,10,12,4,23,4,5,6,9,0,10, 11,21,3,4,5,6,20,34,2,32,42,42, 4,5,6,7,8,9,0,4,3,5,6,7, 2,5,7,4,3,5,6,7,5,3,4];
		let expected = vec![1, 2, 0, 4, 12, 4, 23, 4, 34, 2, 32, 42];
		let mut bindings: Bindings = Bindings::initialize_two(input, memory);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "loadShared_test", 2)));
	}

	#[test]
	fn setConstant_test() {
		let input = vec![0, 0,0,0,0,0,0,0,0,0,0,0];
		let memory = vec![1, 2,0,4,6,8,2,3,4,5,2,4,4, 5,6,10,12,4,23,4,5,6,9,0,10, 11,21,3,4,5,6,20,34,2,32,42,42, 4,5,6,7,8,9,0,4,3,5,6,7, 2,5,7,4,3,5,6,7,5,3,4];
		let expected = vec![12, 4, 23, 4, 12, 4, 23, 4, 12, 4, 23, 4];
		let mut bindings: Bindings = Bindings::initialize_two(input, memory);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "setConstant_test", 2)));
	}

	#[test]
	fn setZero_test() {
		let input = vec![0, 0,0,0,0,0,0,0,0,0,0,0];
		let memory = vec![1, 2,0,4,6,8,2,3,4,5,2,4,4, 5,6,10,12,4,23,4,5,6,9,0,10, 11,21,3,4,5,6,20,34,2,32,42,42, 4,5,6,7,8,9,0,4,3,5,6,7, 2,5,7,4,3,5,6,7,5,3,4];
		let expected = vec![1, 2, 0, 4, 1, 2, 0, 4, 1, 2, 0, 4];
		let mut bindings: Bindings = Bindings::initialize_two(input, memory);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "setZero_test", 2)));
	}

	#[test]
	fn setN_test() {
		let input = vec![0, 0,0,0,0,0,0,0,0,0,0,0];
		let memory = vec![1, 2,0,4,6,8,2,3,4,5,2,4,4, 5,6,10,12,4,23,4,5,6,9,0,10, 11,21,3,4,5,6,20,34,2,32,42,42, 4,5,6,7,8,9,0,4,3,5,6,7, 2,5,7,4,3,5,6,7,5,3,4];
		let expected = vec![12, 4, 23, 4, 12, 4, 23, 4, 12, 4, 23, 4];
		let mut bindings: Bindings = Bindings::initialize_two(input, memory);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "setN_test", 2)));
	}

	#[test]
	fn setOne_test() {
		let input = vec![0, 0,0,0,0,0,0,0,0,0,0,0];
		let memory = vec![1, 2,0,4,6,8,2,3,4,5,2,4,4, 5,6,10,12,4,23,4,5,6,9,0,10, 11,21,3,4,5,6,20,34,2,32,42,42, 4,5,6,7,8,9,0,4,3,5,6,7, 2,5,7,4,3,5,6,7,5,3,4];
		let expected = vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
		let mut bindings: Bindings = Bindings::initialize_two(input, memory);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "setOne_test", 2)));
	}

	#[test]
	fn setRSquared_test() {
		let input = vec![0, 0,0,0,0,0,0,0,0,0,0,0];
		let memory = vec![1, 2,0,4,6,8,2,3,4,5,2,4,4, 5,6,10,12,4,23,4,5,6,9,0,10, 11,21,3,4,5,6,20,34,2,32,42,42, 4,5,6,7,8,9,0,4,3,5,6,7, 2,5,7,4,3,5,6,7,5,3,4];
		let expected = vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
		let mut bindings: Bindings = Bindings::initialize_two(input, memory);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "setRSquared_test", 2)));
	}

	#[test]
	fn setPermuteLow_test() {
		let input = vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
		let expected = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
		let mut bindings: Bindings = Bindings::initialize_one(input);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "setPermuteLow_test", 1)));
	}

	#[test]
	fn setPermuteHigh_test() {
		let input = vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
		let expected = vec![0, 0, 0, 0, 4, 4, 4, 4, 0, 0, 0, 0];
		let mut bindings: Bindings = Bindings::initialize_one(input);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "setPermuteHigh_test", 1)));
	}

	#[test]
	fn copyToShared_test() {
		let input = vec![0; 300];
		let expected = vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
		let shared_memory = vec![0; 300];
		
		let mut bindings: Bindings = Bindings::initialize_two(input, shared_memory);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "copyToShared_test", 2)));
	}

	#[test]
	fn copyTocopyCountsAndIndexes_test() {
		let input = vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
		let expected = vec![1, 8, 3, 16, 1, 8, 3, 16, 1, 8, 3, 16, 12];
		let shared_memory = vec![0; 96];
		
		let mut bindings: Bindings = Bindings::initialize_two(input, shared_memory);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "copyCountsAndIndexes_test", 2)));
	}

	#[test]
	fn copyPointIndexes_test() {
		let input = vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
		let expected = vec![4, 2, 4, 4, 4, 3, 4, 4, 4, 4, 4, 4, 4, 1, 4, 4, 4, 2, 4, 4, 4, 3, 4, 4, 0, 0, 0, 0];
		let shared_memory = vec![4; 381];
		
		let mut bindings: Bindings = Bindings::initialize_two(input, shared_memory);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "copyPointIndexes_test", 2)));
	}

	#[test]
	fn prefetch_test() {
		let input = vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
		let expected = vec![2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2];
		let shared_memory = vec![2; 96];
		
		let mut bindings: Bindings = Bindings::initialize_two(input, shared_memory);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "prefetch_test", 2)));
	}

	#[test]
	fn computeBucketSums_test() {
		let input = vec![100; 48];
		let expected = vec![0; 48];
		let shared_memory = vec![100; 96];
		
		let mut bindings: Bindings = Bindings::initialize_two(input, shared_memory);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "computeBucketSums_test", 2)));
	}

	#[test]
	fn slice23_test() {
		let input = vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
		let expected = vec![100, 0, 0, 3, 0, 0, 0, 50, 0, 0, 1, 0];

		
		let mut bindings: Bindings = Bindings::initialize_one(input);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "slice23_test", 1)));
	}

	#[test]
	fn sub_psd_test() {
		let input = vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
		let expected = vec![99, 99, 99, 99, 99, 99, 99, 99, 0, 0, 0, 0, 0];
		
		let mut bindings: Bindings = Bindings::initialize_one(input);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "sub_psd_test", 1)));
	}

	#[test]
	fn addN_psd_test() {
		let input = vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
		let expected = vec![101, 168919140, 3489661029, 1504343906, 1547153509, 1622429058, 2586617274, 313222594, 0, 0, 0, 0, 0];
		
		let mut bindings: Bindings = Bindings::initialize_one(input);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "addN_psd_test", 1)));
	}

	#[test]
	fn negate_test() {
		let input = vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
		let expected = vec![4294967197, 168918940, 3489660829, 1504343706, 1547153309, 1622428858, 2586617074, 313222394, 0, 0, 0, 0, 0];
		
		let mut bindings: Bindings = Bindings::initialize_one(input);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "negate_test", 1)));
	}

	#[test]
	fn ballot_sync_test() {
		let input = vec![100];
		let expected = vec![0];
		
		let mut bindings: Bindings = Bindings::initialize_one(input);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "ballot_sync_test", 1)));
	}

	#[test]
	fn processSignedDigitsKernel_test() {
		let input = vec![0; 300];
		let expected = vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 168919040, 337838080, 506757120, 675676160, 844595200, 1013514240, 1182433280, 1351352320, 1520271360, 1689190400, 1858109440, 2027028480, 2195947520, 2364866560, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
		let shared_memory = vec![1; 300];

		let mut bindings: Bindings = Bindings::initialize_two(input, shared_memory);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "processSignedDigitsKernel_test", 2)));
	}

	#[test]
	fn sqr1_test() {
		let input = vec![0; 12];
		let expected = vec![771999968, 1049258088, 1049258088, 1049258088, 1049258088, 1049258088, 1049258088, 1049258088, 1049258088, 1049258088, 1049258088, 277258120];

		let mut bindings: Bindings = Bindings::initialize_one(input);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "sqr1_test", 1)));
	}

	#[test]
	fn mul1_test() {
		let input = vec![0; 12];
		let expected = vec![401387532, 2766392126, 2766392126, 2766392126, 2766392126, 2766392126, 2766392126, 2766392126, 2766392126, 2766392126, 2766392126, 2365004594];

		let mut bindings: Bindings = Bindings::initialize_one(input);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "mul1_test", 1)));
	}

	#[test]
	fn mul2_test() {
		let input = vec![0; 24];
		let expected = vec![401387532, 401387532, 401387532, 401387532, 401387532, 401387532, 401387532, 401387532, 401387532, 401387532, 401387532, 401387532, 
																	2365004594, 2365004594, 2365004594, 2365004594, 2365004594, 2365004594, 2365004594, 2365004594, 2365004594, 2365004594, 2365004594, 2365004594];

		let mut bindings: Bindings = Bindings::initialize_one(input);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "mul2_test", 1)));
	}

	#[test]
	fn sqr2_test() {
		let input = vec![0; 24];
		let expected = vec![737864440, 737864440, 737864440, 737864440, 737864440, 737864440, 737864440, 737864440, 737864440, 737864440, 737864440, 737864440, 
																	2365004594, 2365004594, 2365004594, 2365004594, 2365004594, 2365004594, 2365004594, 2365004594, 2365004594, 2365004594, 2365004594, 2365004594];

		let mut bindings: Bindings = Bindings::initialize_one(input);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "sqr2_test", 1)));
	}

	#[test]
	fn merge_test() {
		let input = vec![0; 12];
		let expected = vec![100, 200, 203, 102, 101, 102, 101, 102, 101, 102, 101, 102];

		let mut bindings: Bindings = Bindings::initialize_one(input);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "merge_test", 1)));
	}

	#[test]
	fn add1_HighThroughput_test() {
		let input = vec![0; 48];
		let expected = vec![100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 
																	99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 
																	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
																	100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100];
		let shared_memory = vec![0; 96];

		let mut bindings: Bindings = Bindings::initialize_two(input, shared_memory);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "add1_HighThroughput_test", 2)));
	}

	#[test]
	fn add2_HighThroughput_test() {
		let input = vec![0; 48];
		let expected = vec![100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 
																	99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 0, 
																	100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 99, 
																	99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99];
		let shared_memory = vec![0; 96];

		let mut bindings: Bindings = Bindings::initialize_two(input, shared_memory);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "add2_HighThroughput_test", 2)));
	}

	#[test]
	fn dbl_test() {
		let input = vec![0; 48];
		let expected = vec![1451445187, 2674199856, 3525627233, 2269570348, 1883284832, 2667508719, 1157892109, 2425183033, 2282268945, 2497044129, 2306080800, 2935882546, 
																	2024533096, 208261594, 2272788712, 1918972986, 2243500812, 2282208119, 3877677448, 4060338964, 3294418192, 3420973674, 3637195670, 1972927228, 
																	3838154432, 2424424896, 2375453312, 2375453312, 2375453312, 2375453312, 2375453312, 2375453312, 2375453312, 2375453312, 2375453312, 2098512064, 
																	4251270144, 1703055117, 2637511309, 2774372109, 3170858125, 3170858125, 3170858125, 3170858125, 3170858125, 3170858125, 3170858125, 2375810701];
		let shared_memory = vec![0; 96];

		let mut bindings: Bindings = Bindings::initialize_two(input, shared_memory);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "dbl_test", 2)));
	}

	#[test]
	fn accumulator_test() {
		let input = vec![0; 48];
		let expected = vec![100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 
																	99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 
																	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
																	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
		let shared_memory = vec![0; 96];

		let mut bindings: Bindings = Bindings::initialize_two(input, shared_memory);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "accumulator_test", 2)));
	}

	#[test]
	fn normalize_test() {
		let input = vec![0; 49];
		let expected = vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
																	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
																	100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 
																	99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 0];
		let shared_memory = vec![0; 96];

		let mut bindings: Bindings = Bindings::initialize_two(input, shared_memory);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "normalize_test", 2)));
	}

	#[test]
	fn fromInternal_HighThroughput_test() {
		let input = vec![0; 48];
		let expected = vec![3903961325, 1049258089, 4181219445, 4181219445, 4181219445, 4181219445, 4181219445, 4181219445, 4181219445, 4181219445, 4181219445, 3409219477, 
																	3166096885, 2241356351, 1078350411, 1078350411, 1078350411, 1078350411, 1078350411, 1078350411, 1078350411, 1078350411, 1078350411, 1044214883, 
																	3903961325, 1049258089, 4181219445, 4181219445, 4181219445, 4181219445, 4181219445, 4181219445, 4181219445, 4181219445, 4181219445, 3409219477, 
																	3166096885, 2241356351, 1078350411, 1078350411, 1078350411, 1078350411, 1078350411, 1078350411, 1078350411, 1078350411, 1078350411, 1044214883];
		let shared_memory = vec![0; 96];

		let mut bindings: Bindings = Bindings::initialize_two(input, shared_memory);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "fromInternal_HighThroughput_test", 2)));
	}

	#[test]
	fn initializeCountersSizesAtomicsHistogramKernel_test() {
		let input = vec![0; 60];
		let expected = vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
																	11257, 11258, 11259, 11260, 11261, 11262, 11263, 11251, 11252, 11253, 11254, 11256, 
																	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
																	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
																	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

		let mut bindings: Bindings = Bindings::initialize_one(input);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "initializeCountersSizesAtomicsHistogramKernel_test", 1)));
	}

	#[test]
	fn sizesPrefixSumKernel_test() {
		let input = vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
		let expected = vec![0, 0, 6348, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 6348, 1, 1, 1, 1, 1, 1, 1, 1, 1];
		let shared_memory = vec![0; 96];

		let mut bindings: Bindings = Bindings::initialize_two(input, shared_memory);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "sizesPrefixSumKernel_test", 2)));
	}

	#[test]
	fn warpPrefixSum_test() {
		let input = vec![0];
		let expected = vec![2000];

		let mut bindings: Bindings = Bindings::initialize_one(input);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "warpPrefixSum_test", 1)));
	}

	#[test]
	fn multiwarpPrefixSum1_test() {
		let input = vec![0];
		let expected = vec![4032];

		let mut bindings: Bindings = Bindings::initialize_one(input);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "multiwarpPrefixSum1_test", 1)));
	}

	#[test]
	fn multiwarpPrefixSum2_test() {
		let input = vec![0];
		let expected = vec![4300];
		let shared_memory = vec![100; 96]; 

		let mut bindings: Bindings = Bindings::initialize_two(input, shared_memory);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "multiwarpPrefixSum2_test", 2)));
	}

	#[test]
	fn udiv3_test() {
		let input = vec![100];
		let expected = vec![1431655832];

		let mut bindings: Bindings = Bindings::initialize_one(input);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "udiv3_test", 1)));
	}

	#[test]
	fn udiv5_test() {
		let input = vec![100];
		let expected = vec![80];

		let mut bindings: Bindings = Bindings::initialize_one(input);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "udiv5_test", 1)));
	}

	#[test]
	fn compress_test() {
		let input = vec![100];
		let expected = vec![25700];

		let mut bindings: Bindings = Bindings::initialize_one(input);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "compress_test", 1)));
	}

	#[test]
	fn precomputePointsKernel_test() {
		let input = vec![0; 24];
		let expected = vec![1715505673, 3946063407, 3923780668, 2493396589, 2351260709, 3821750944, 3066524883, 4292347058, 
																	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
		let shared_memory = vec![0; 96];

		let mut bindings: Bindings = Bindings::initialize_two(input, shared_memory);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "precomputePointsKernel_test", 2)));
	}

	#[test]
	fn nextPage_test() {
		let input = vec![0];
		let expected = vec![200];

		let mut bindings: Bindings = Bindings::initialize_one(input);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "nextPage_test", 1)));
	}

	#[test]
	fn initializeShared_test() {
		let input = vec![1; 200];
		let expected = vec![0; 200];
		let shared = vec![0; 200];

		let mut bindings: Bindings = Bindings::initialize_two(input, shared);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "initializeShared_test", 2)));
	}

	#[test]
	fn clz_test() {
		let input = vec![0];
		let expected = vec![28];

		let mut bindings: Bindings = Bindings::initialize_one(input);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "clz_test", 1)));
	}

	#[test]
	fn cleanup1_test() {
		let input = vec![1; 24];
		let expected = vec![100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200];
		let shared = vec![0; 200];

		let mut bindings: Bindings = Bindings::initialize_two(input, shared);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "cleanup1_test", 2)));
	}

	#[test]
	fn processWrites_test() {
		let input = vec![1; 24];
		let expected = vec![100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200];
		let shared = vec![0; 200];

		let mut bindings: Bindings = Bindings::initialize_two(input, shared);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "processWrites_test", 2)));
	}

	#[test]
	fn partition1024Kernel_test() {
		let input = vec![1; 24];
		let expected = vec![100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200];
		let shared = vec![0; 200];

		let mut bindings: Bindings = Bindings::initialize_two(input, shared);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "partition1024Kernel_test", 2)));
	}

	#[test]
	fn round128_test() {
		let input = vec![1];
		let expected = vec![128];

		let mut bindings: Bindings = Bindings::initialize_one(input);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "round128_test", 1)));
	}

	#[test]
	fn initializeShared4096_test() {
		let input = vec![0; 300];
		let expected = vec![0, 0, 0, 0, 0, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
		let shared = vec![0; 300];

		let mut bindings: Bindings = Bindings::initialize_two(input, shared);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "initializeShared4096_test", 2)));
	}

	#[test]
	fn read640_test() {
		let input = vec![0; 24];
		let expected = vec![5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 0, 0, 0, 0, 0, 0];

		let mut bindings: Bindings = Bindings::initialize_one(input);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "read640_test", 1)));
	}

	#[test]
	fn shared_copy_u4_test() {
		let input = vec![0; 12];
		let expected = vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
		let shared = vec![1; 300];

		let mut bindings: Bindings = Bindings::initialize_two(input, shared);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "shared_copy_u4_test", 2)));
	}

	#[test]
	fn prefixSumBuckets_test() {
		let input = vec![0; 300];
		let expected = vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 102, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 101, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 101, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 100, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
		1, 1, 1, 100, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
		let shared = vec![1; 300];

		let mut bindings: Bindings = Bindings::initialize_two(input, shared);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "prefixSumBuckets_test", 2)));
	}

	#[test]
	fn sortMap_test() {
		let input = vec![0; 300];
		let expected = vec![10; 300];
		let shared = vec![10; 300];

		let mut bindings: Bindings = Bindings::initialize_two(input, shared);
		assert_ne!(expected, pollster::block_on(run(&mut bindings, "sortMap_test", 2)));
	}

	#[test]
	fn unpackData_test() {
		let input = vec![0; 8];
		let expected = vec![39, 78, 1171, 0, 39, 78, 19, 0];

		let mut bindings: Bindings = Bindings::initialize_one(input);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "unpackData_test", 1)));
	}

	#[test]
	fn cleanupShared_test() {
		let input = vec![0; 300];
		let expected = vec![0, 10, 10, 10, 0, 10, 10, 10, 0, 10, 10, 10, 0, 10, 10, 10, 0, 10, 10, 10, 0, 10, 10, 10, 0, 10, 10, 10, 0, 10, 10, 10, 0, 10, 10, 10, 0, 10, 10, 
		10, 0, 10, 10, 10, 0, 10, 10, 10, 0, 10, 10, 10, 0, 10, 10, 10, 0, 10, 10, 10, 0, 10, 10, 10, 0, 10, 10, 10, 0, 10, 10, 10, 0, 10, 10, 10, 0, 10, 10, 10, 0, 10, 10, 10, 0, 10, 10, 10, 0, 10, 10, 10, 0, 10, 10, 10, 0, 10, 10, 10, 0, 10, 10, 10, 0, 10, 10, 10, 0, 10, 10, 10, 0, 10, 10, 10, 0, 10, 10, 10, 0, 10, 10, 10, 0, 10, 10, 10, 0, 10, 10, 10, 0, 10, 10, 10, 0, 10, 10, 10, 0, 10, 10, 10, 0, 10, 10, 10, 0, 10, 10, 10, 0, 10, 10, 10, 0, 10, 10, 10, 0, 10, 10, 10, 0, 10, 10, 10, 0, 10, 10, 10, 0, 10, 10, 10, 0, 10, 10, 10, 0, 10, 10, 10, 0, 10, 10, 10, 0, 10, 10, 10, 0, 10, 10, 10, 0, 10, 10, 10, 0, 10, 10, 10, 0, 10, 10, 10, 0, 10, 10, 10, 0, 10, 10, 10, 0, 10, 10, 10, 0, 10, 10, 10, 0, 10, 10, 10, 0, 10, 10, 10, 0, 10, 10, 10, 0, 10, 10, 10, 0, 10, 10, 10, 0, 10, 10, 
		10, 0, 10, 10, 10, 0, 10, 10, 10, 0, 10, 10, 10, 0, 10, 10, 10, 0, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10];
		let shared = vec![10; 300];

		let mut bindings: Bindings = Bindings::initialize_two(input, shared);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "cleanupShared_test", 2)));
	}

	#[test]
	fn writePointToShared_test() {
		let input = vec![0; 300];
		let expected = vec![10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 1, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 14, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10];
		let shared = vec![10; 300];

		let mut bindings: Bindings = Bindings::initialize_two(input, shared);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "writePointToShared_test", 2)));
	}

	#[test]
	fn partitionPagesToScratch_test() {
		let input = vec![0; 300];
		let expected = vec![10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 0, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 0, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 0, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 0, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 0, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 0, 10, 10, 10, 10, 
		10, 10, 10, 10, 10, 10, 10, 0, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 0, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 0, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 0, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 0, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 0, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 0, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 0, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 0, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 0, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 0, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 0, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 0, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 0, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 0, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 0, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 
		10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10];
		let shared = vec![10; 300];

		let mut bindings: Bindings = Bindings::initialize_two(input, shared);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "partitionPagesToScratch_test", 2)));
	}

	#[test]
	fn partitionScratchToPoints_test() {
		let input = vec![0; 12];
		let expected = vec![10, 2, 3, 4, 5, 10, 7, 8, 9, 10, 11, 12];
		let shared = vec![10; 300];

		let mut bindings: Bindings = Bindings::initialize_two(input, shared);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "partitionScratchToPoints_test", 2)));
	}

	#[test]
	fn countFromPages_test() {
		let input = vec![0; 300];
		let expected = vec![10; 300];
		let shared = vec![10; 300];

		let mut bindings: Bindings = Bindings::initialize_two(input, shared);
		assert_ne!(expected, pollster::block_on(run(&mut bindings, "countFromPages_test", 2)));
	}

	#[test]
	fn partitionPagesToPoints_test() {
		let input = vec![0; 12];
		let expected = vec![0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
		let shared = vec![10; 300];

		let mut bindings: Bindings = Bindings::initialize_two(input, shared);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "partitionPagesToPoints_test", 2)));
	}

	#[test]
	fn partition4096Kernel_test() {
		let input = vec![0; 300];
		let expected = vec![100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 0, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 0, 100, 100, 100, 
		100, 100, 100, 100, 100, 100, 100, 100, 0, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 0, 100, 100, 100, 100, 100, 100, 100, 100, 100, 
		100, 100, 0, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 0, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 0, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 0, 100, 100, 0, 100, 100, 100, 100, 100, 100, 100, 100, 0, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 0, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 0, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 0, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 0, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 0, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 0, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 0, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 0, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 0, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 0, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 0, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 0, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 0, 100, 100, 100, 100, 100, 
		100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100];
		let shared = vec![100; 300];

		let mut bindings: Bindings = Bindings::initialize_two(input, shared);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "partition4096Kernel_test", 2)));
	}

	#[test]
	fn partition1024Kernel_call_test() {
		let input = vec![1, 1024, 65536, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11008, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 65536, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
		let expected = vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 168919040, 337838080, 506757120, 675676160, 844595200, 1013514240, 1182433280, 1351352320, 1520271360, 1689190400, 1858109440, 2027028480, 2195947520, 2364866560, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
		0, 0, 0];
		let shared = vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 168919040, 337838080, 506757120, 675676160, 844595200, 1013514240, 1182433280, 1351352320, 1520271360, 1689190400, 1858109440, 2027028480, 2195947520, 2364866560, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

		let mut bindings: Bindings = Bindings::initialize_two(input, shared);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "partition1024Kernel_call_test", 2)));
	}

	#[test]
	fn histogramPrefixSumKernel_test() {
		let input = vec![0; 12];
		let expected = vec![1, 200, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
		
		let mut bindings: Bindings = Bindings::initialize_one(input);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "histogramPrefixSumKernel_test", 1)));
	}

	#[test]
	fn sortCountsKernel_test() {
		let input = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
		let expected = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
		let shared = vec![1; 300];
		
		let mut bindings: Bindings = Bindings::initialize_two(input, shared);
		assert_eq!(expected, pollster::block_on(run(&mut bindings, "sortCountsKernel_test", 2)));
	}