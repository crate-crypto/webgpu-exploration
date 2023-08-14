#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
#![allow(dead_code)]

use std::{collections::HashMap, fs};

use toml::Value;
use wgpu::{util::DeviceExt, BindGroupEntry, Instance, Device, Queue, AdapterInfo, ShaderModule, Buffer, Backends};

mod MSM;
mod host_reduce;
mod host_curve;
mod utility;
mod batch_functions;
mod MSM_cpu;
mod grafics;

pub use MSM::{MSMReadHexPoints, MSMReadHexScalars};
pub use utility::{u32_as_mut_slice_u8, u8_as_mut_slice_u32, u32_as_mut_slice_u64, u8_as_slice_u32};
pub use host_curve::*;
pub use batch_functions::*;
pub use MSM_cpu::*;
pub use grafics::*;

pub use crate::MSM::MSMContext;

macro_rules! all_files {
	($($file:expr),*) => {
		{String::new()$(+include_str!($file)+"\n")*}
	};
}

fn read_config_file(file_path: &str) -> Result<HashMap<String, Value>, Box<dyn std::error::Error>> {
	let contents = fs::read_to_string(file_path)?;

	let config: HashMap<String, Value> = toml::from_str(&contents)?;

	Ok(config)
}

// Indicates a u32 overflow in an intermediate Collatz value
const OVERFLOW: u32 = 0xffffffff;

pub async fn run(numbers: &mut Bindings, func_name: &str, binding_number: u32) -> Vec<u32> {
	let steps = execute_gpu(numbers, func_name, binding_number).await.unwrap();

	// let disp_steps: Vec<String> = steps
	// 		.iter()
	// 		.map(|&n| match n {
	// 				OVERFLOW => "OVERFLOW".to_string(),
	// 				_ => n.to_string(),
	// 		})
	// 		.collect();

	//println!("Steps: [{}]", disp_steps.join(", "));
	// #[cfg(target_arch = "wasm32")]
	// log::info!("Steps: [{}]", disp_steps.join(", "));
	steps
}

pub struct BufCoder {
	staging_buffer: Buffer,
}

impl BufCoder {
	pub fn initialize(gpu: &GpuConsts, numbers: &mut Bindings, func_name: &str, binding_number: u32) -> BufCoder {
		let file_path = "config.toml";
		// Gets the size in bytes of the buffer.
		let slice_size = numbers.input_output.len() * std::mem::size_of::<u32>();
		let size = slice_size as wgpu::BufferAddress;

		// Instantiates buffer without data.
		// `usage` of buffer specifies how it can be used:
		//   `BufferUsages::MAP_READ` allows it to be read (outside the shader).
		//   `BufferUsages::COPY_DST` allows it to be the destination of the copy.
		let staging_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
				label: None,
				size,
				usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
				mapped_at_creation: false,
		});

		// Instantiates buffer with data (`numbers`).
		// Usage allowing the buffer to be:
		//   A storage buffer (can be bound within a bind group and thus available to a shader).
		//   The destination of a copy.
		//   The source of a copy.
		let storage_buffer = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
				label: Some("Storage Buffer"),
				contents: bytemuck::cast_slice(&numbers.input_output),
				usage: wgpu::BufferUsages::STORAGE
						| wgpu::BufferUsages::COPY_DST
						| wgpu::BufferUsages::COPY_SRC,
		});

		// A bind group defines how buffers are accessed by shaders.
		// It is to WebGPU what a descriptor set is to Vulkan.
		// `binding` here refers to the `binding` of a buffer in the shader (`layout(set = 0, binding = 0) buffer`).

		// A pipeline specifies the operation of a shader

		// Instantiates the pipeline.
		let compute_pipeline = gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
				label: None,
				layout: None,
				module: &gpu.cs_module,
				entry_point: func_name,
		});

		// Instantiates the bind group, once again specifying the binding of buffers.
		let bind_group_layout = compute_pipeline.get_bind_group_layout(read_config_file(file_path).unwrap().get("layout").unwrap().as_integer().unwrap() as u32);

		let mut new_binding_entries: Vec<BindGroupEntry> = vec![
		wgpu::BindGroupEntry {
			binding: 0,
			resource: storage_buffer.as_entire_binding(),
		}];

		let storage_buffer2 = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
			label: Some("Shared Memory Buffer"),
			contents: bytemuck::cast_slice(&numbers.shared_memory),
			usage: wgpu::BufferUsages::STORAGE
		});

		let storage_buffer3 = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
			label: Some("Global Memory Buffer"),
			contents: bytemuck::cast_slice(&numbers.global_memory),
			usage: wgpu::BufferUsages::STORAGE
		});

		if binding_number > 1 {
			new_binding_entries.push(
			wgpu::BindGroupEntry {
				binding: 1,
				resource: storage_buffer2.as_entire_binding(),
			});

			if binding_number > 2 {
				new_binding_entries.push(
					wgpu::BindGroupEntry {
						binding: 2,
						resource: storage_buffer3.as_entire_binding(),
					});
			}
		}
		
		let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
			label: None,
			layout: &bind_group_layout,
			entries: &new_binding_entries,
		});

		// A command encoder executes one or many pipelines.
		// It is to WebGPU what a command buffer is to Vulkan.
		let mut encoder =
			gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
		{
			let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
			cpass.set_pipeline(&compute_pipeline);
			cpass.set_bind_group(read_config_file(file_path).unwrap().get("index").unwrap().as_integer().unwrap() as u32, &bind_group, &[]);
			cpass.insert_debug_marker("compute collatz iterations");
			cpass.dispatch_workgroups(
				read_config_file(file_path).unwrap().get("x").unwrap().as_integer().unwrap() as u32,
				read_config_file(file_path).unwrap().get("y").unwrap().as_integer().unwrap() as u32,
				read_config_file(file_path).unwrap().get("z").unwrap().as_integer().unwrap() as u32
			); // Number of cells to run, the (x,y,z) size of item being processed
		}
		// Sets adds copy operation to command encoder.
		// Will copy data from storage buffer on GPU to staging buffer on CPU.
		encoder.copy_buffer_to_buffer(&storage_buffer, 0, &staging_buffer, 0, size);
	
		// Submits command encoder for processing
		gpu.queue.submit(Some(encoder.finish()));

		BufCoder{staging_buffer}
	}
}

pub struct GpuConsts {
	instance: Instance,
	device: Device,
	queue: Queue,
	info: AdapterInfo,
	cs_module: ShaderModule,
}

impl GpuConsts {
	pub async fn initialaze() -> Result<GpuConsts, String> {
		// Instantiates instance of WebGPU
		let instance = wgpu::Instance::default();

		// List of all adapters
    let adapters = instance.enumerate_adapters(Backends::all()).collect::<Vec<_>>();
		

		// Print info about all adapters
    // for adapter in &adapters {
		// 	println!("Adapter name: {:?}", adapter.get_info().name);
		// 	println!("Adapter type: {:?}", adapter.get_info().device_type);
		// 	// Тут можна вивести більше інформації про адаптер, якщо потрібно
		// 	println!();
		// }

		// Find the CPU adapter in the list or default to the first available adapter
    // let selected_adapter_index = adapters
		// .iter()
		// .position(|adapter| adapter.get_info().device_type == wgpu::DeviceType::Cpu)
    // .unwrap_or(0);

    // Use the selected adapter
    let selected_adapter = &adapters[0];

    //println!("Selected adapter: {:?}, type: {:?}", selected_adapter.get_info().name, selected_adapter.get_info().device_type);

		let adapter = selected_adapter;

		//`request_adapter` instantiates the general connection to the GPU
		// let adapter = instance
		// .request_adapter(&wgpu::RequestAdapterOptions::default())
		// .await
		// .ok_or_else(|| "adapter error")?;

		// `request_device` instantiates the feature specific connection to the GPU, defining some parameters,
		//  `features` being the available features.
		let (device, queue) = adapter
				.request_device(
						&wgpu::DeviceDescriptor {
								label: None,
								features: wgpu::Features::empty(),
								limits: wgpu::Limits::downlevel_defaults(),
						},
						None,
				)
				.await
				.unwrap();

		let info = adapter.get_info();

		if info.vendor == 0x10005 {
			return Err("info error".to_string());
		}

		let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
			label: None,
			source: wgpu::ShaderSource::Wgsl(all_files!(
				"curve.wgsl", "../tests/curve_tests.wgsl", "mp.wgsl", "chain.wgsl", 
				"asm.wgsl", "../tests/chain_tests.wgsl", "../tests/mp_tests.wgsl", "compute_bucket_sums.wgsl", 
				"SHM.wgsl", "MSM.wgsl", "../tests/SHM_tests.wgsl", "../tests/compute_bucket_sums_tests.wgsl", 
				"process_signed_digits.wgsl", "support.wgsl", "functions_call.wgsl",
				"../tests/process_signed_digits_tests.wgsl", "initialize_counters_and_sizes.wgsl",
				"../tests/initialize_counters_tests.wgsl", "../tests/support_tests.wgsl", "partition1024.wgsl",
				"partition4096.wgsl", "sort_counts.wgsl", "reduce_buckets.wgsl", "precompute_points.wgsl",
				"../tests/precompute_points_tests.wgsl", "../tests/partition1024_tests.wgsl", "../tests/partition4096_tests.wgsl",
				"../tests/sort_counts_test.wgsl", "curve_functions_call.wgsl", "batch_functions_call.wgsl"
			).into()),
		});

		Ok(GpuConsts{instance, device, queue, info, cs_module})
	}

	pub async fn run(&self, bufcoder: &BufCoder) -> Option<Vec<u32>> {
		// Note that we're not calling `.await` here.
		let buffer_slice = bufcoder.staging_buffer.slice(..);
		// Sets the buffer up for mapping, sending over the result of the mapping back to us when it is finished.
		let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
		buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

		// Poll the device in a blocking manner so that our future resolves.
		// In an actual application, `device.poll(...)` should
		// be called in an event loop or on another thread.
		self.device.poll(wgpu::Maintain::Wait);

		// Awaits until `buffer_future` can be read from
		if let Some(Ok(())) = receiver.receive().await {
		// Gets contents of buffer
		let data = buffer_slice.get_mapped_range();
		// Since contents are got in bytes, this converts these bytes back to u32
		let result = bytemuck::cast_slice(&data).to_vec();

		// With the current interface, we have to make sure all mapped views are
		// dropped before we unmap the buffer.
		drop(data);
		bufcoder.staging_buffer.unmap(); // Unmaps buffer from memory
														// If you are familiar with C++ these 2 lines can be thought of similarly to:
														//   delete myPointer;
														//   myPointer = NULL;
														// It effectively frees the memory
		
		// Returns data from buffer
		Some(result)
		} else {
			panic!("failed to run compute on gpu!")
		}
	}
}

async fn execute_gpu(numbers: &mut Bindings, func_name: &str, binding_number: u32) -> Option<Vec<u32>> {
	// Instantiates instance of WebGPU
	let instance = wgpu::Instance::default();

	// `request_adapter` instantiates the general connection to the GPU
	let adapter = instance
			.request_adapter(&wgpu::RequestAdapterOptions::default())
			.await?;

	// `request_device` instantiates the feature specific connection to the GPU, defining some parameters,
	//  `features` being the available features.
	let (device, queue) = adapter
			.request_device(
					&wgpu::DeviceDescriptor {
							label: None,
							features: wgpu::Features::empty(),
							limits: wgpu::Limits::downlevel_defaults(),
					},
					None,
			)
			.await
			.unwrap();

	let info = adapter.get_info();
	// skip this on LavaPipe temporarily
	if info.vendor == 0x10005 {
			return None;
	}

	execute_gpu_inner(&device, &queue, numbers, func_name, binding_number).await
}

pub struct Bindings {
	input_output: Vec<u32>,
	shared_memory: Vec<u32>,
	global_memory: Vec<u32>,
}

impl Bindings {
	pub fn initialize_one(input_output: Vec<u32>) -> Self {
		Bindings{input_output, shared_memory: <_>::default(), global_memory: <_>::default()}
	}

	pub fn initialize_two(input_output: Vec<u32>, shared_memory: Vec<u32>) -> Self {
		Bindings{input_output, shared_memory, global_memory: <_>::default()}
	}

	pub fn initialize_three(input_output: Vec<u32>, shared_memory: Vec<u32>, global_memory: Vec<u32>) -> Self {
		Bindings{input_output, shared_memory, global_memory}
	}
}

async fn execute_gpu_inner(
	device: &wgpu::Device,
	queue: &wgpu::Queue,
	numbers: &mut Bindings,
	func_name: &str,
	binding_number: u32,
) -> Option<Vec<u32>> {
	// Loads the shader from WGSL
	let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
			label: None,
			source: wgpu::ShaderSource::Wgsl(all_files!(
				"curve.wgsl", "../tests/curve_tests.wgsl", "mp.wgsl", "chain.wgsl", 
				"asm.wgsl", "../tests/chain_tests.wgsl", "../tests/mp_tests.wgsl", "compute_bucket_sums.wgsl", 
				"SHM.wgsl", "MSM.wgsl", "../tests/SHM_tests.wgsl", "../tests/compute_bucket_sums_tests.wgsl", 
				"process_signed_digits.wgsl", "support.wgsl", "functions_call.wgsl",
				"../tests/process_signed_digits_tests.wgsl", "initialize_counters_and_sizes.wgsl",
				"../tests/initialize_counters_tests.wgsl", "../tests/support_tests.wgsl", "partition1024.wgsl",
				"partition4096.wgsl", "sort_counts.wgsl", "reduce_buckets.wgsl", "precompute_points.wgsl",
				"../tests/precompute_points_tests.wgsl", "../tests/partition1024_tests.wgsl", "../tests/partition4096_tests.wgsl",
				"../tests/sort_counts_test.wgsl", "curve_functions_call.wgsl", "batch_functions_call.wgsl"
			).into()),
	});
	

	// Gets the size in bytes of the buffer.
	let slice_size = numbers.input_output.len() * std::mem::size_of::<u32>();
	let size = slice_size as wgpu::BufferAddress;

	// Instantiates buffer without data.
	// `usage` of buffer specifies how it can be used:
	//   `BufferUsages::MAP_READ` allows it to be read (outside the shader).
	//   `BufferUsages::COPY_DST` allows it to be the destination of the copy.
	let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
			label: None,
			size,
			usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
			mapped_at_creation: false,
	});

	// Instantiates buffer with data (`numbers`).
	// Usage allowing the buffer to be:
	//   A storage buffer (can be bound within a bind group and thus available to a shader).
	//   The destination of a copy.
	//   The source of a copy.
	let storage_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
			label: Some("Storage Buffer"),
			contents: bytemuck::cast_slice(&numbers.input_output),
			usage: wgpu::BufferUsages::STORAGE
					| wgpu::BufferUsages::COPY_DST
					| wgpu::BufferUsages::COPY_SRC,
	});

	// A bind group defines how buffers are accessed by shaders.
	// It is to WebGPU what a descriptor set is to Vulkan.
	// `binding` here refers to the `binding` of a buffer in the shader (`layout(set = 0, binding = 0) buffer`).

	// A pipeline specifies the operation of a shader

	// Instantiates the pipeline.
	let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
			label: None,
			layout: None,
			module: &cs_module,
			entry_point: func_name,
	});

	// Instantiates the bind group, once again specifying the binding of buffers.
	let bind_group_layout = compute_pipeline.get_bind_group_layout(0);

	let mut new_binding_entries: Vec<BindGroupEntry> = vec![
	wgpu::BindGroupEntry {
		binding: 0,
		resource: storage_buffer.as_entire_binding(),
	}];

	let storage_buffer2 = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
		label: Some("Shared Memory Buffer"),
		contents: bytemuck::cast_slice(&numbers.shared_memory),
		usage: wgpu::BufferUsages::STORAGE
	});

	// let storage_buffer3 = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
	// 	label: Some("Global Memory Buffer"),
	// 	contents: bytemuck::cast_slice(&numbers.global_memory),
	// 	usage: wgpu::BufferUsages::STORAGE
	// });

	if binding_number > 1 {
		new_binding_entries.push(
		wgpu::BindGroupEntry {
			binding: 1,
			resource: storage_buffer2.as_entire_binding(),
		});

		// if binding_number > 2 {
		// 	new_binding_entries.push(
		// 		wgpu::BindGroupEntry {
		// 			binding: 2,
		// 			resource: storage_buffer3.as_entire_binding(),
		// 		});
		// }
	}
	
	let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
		label: None,
		layout: &bind_group_layout,
		entries: &new_binding_entries,
	});

	// A command encoder executes one or many pipelines.
	// It is to WebGPU what a command buffer is to Vulkan.
	let mut encoder =
			device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
	{
			let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
			cpass.set_pipeline(&compute_pipeline);
			cpass.set_bind_group(0, &bind_group, &[]);
			cpass.insert_debug_marker("compute collatz iterations");
			cpass.dispatch_workgroups(16, 1, 1); // Number of cells to run, the (x,y,z) size of item being processed
	}
	// Sets adds copy operation to command encoder.
	// Will copy data from storage buffer on GPU to staging buffer on CPU.
	encoder.copy_buffer_to_buffer(&storage_buffer, 0, &staging_buffer, 0, size);

	// Submits command encoder for processing
	queue.submit(Some(encoder.finish()));
	
	// Note that we're not calling `.await` here.
	let buffer_slice = staging_buffer.slice(..);
	// Sets the buffer up for mapping, sending over the result of the mapping back to us when it is finished.
	let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
	buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

	// Poll the device in a blocking manner so that our future resolves.
	// In an actual application, `device.poll(...)` should
	// be called in an event loop or on another thread.
	device.poll(wgpu::Maintain::Wait);

	// Awaits until `buffer_future` can be read from
	if let Some(Ok(())) = receiver.receive().await {
	// Gets contents of buffer
	let data = buffer_slice.get_mapped_range();
	// Since contents are got in bytes, this converts these bytes back to u32
	let result = bytemuck::cast_slice(&data).to_vec();

	// With the current interface, we have to make sure all mapped views are
	// dropped before we unmap the buffer.
	drop(data);
	staging_buffer.unmap(); // Unmaps buffer from memory
													// If you are familiar with C++ these 2 lines can be thought of similarly to:
													//   delete myPointer;
													//   myPointer = NULL;
													// It effectively frees the memory

	// Returns data from buffer
	Some(result)
	} else {
		panic!("failed to run compute on gpu!")
	}
}