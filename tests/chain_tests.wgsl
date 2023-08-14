@compute
@workgroup_size(1)
fn initialize_chain_t1_test() {
  var chain: chain_t;

  initialize_chain_t1(&chain);
  v_indices[0] = u32(chain.firstOperation);
}

@compute
@workgroup_size(1)
fn initialize_chain_t2_test() {
  var chain: chain_t;

  initialize_chain_t2(&chain, bool(v_indices[0]));
  v_indices[0] = u32(chain.firstOperation);
}

@compute
@workgroup_size(1)
fn reset1_test() {
  var chain: chain_t;

  reset1(&chain);
  v_indices[0] = u32(chain.firstOperation);
}

@compute
@workgroup_size(1)
fn reset2_test() {
  var chain: chain_t;

  reset2(&chain, bool(v_indices[0]));
  v_indices[0] = u32(chain.firstOperation);
}

@compute
@workgroup_size(1)
fn getCarry_test() {
  var chain: chain_t;
  
  v_indices[0] = u32(getCarry(&chain));
}

@compute
@workgroup_size(1)
fn add_chain_t_test() {
  var chain: chain_t;
  
  v_indices[0] = u32(add_chain_t(&chain, v_indices[0], v_indices[1]));
  v_indices[1] = u32(chain.firstOperation);
}

@compute
@workgroup_size(1)
fn sub_chain_t_test() {
  var chain: chain_t;
  
  v_indices[0] = u32(sub_chain_t(&chain, v_indices[0], v_indices[1]));
  v_indices[1] = u32(chain.firstOperation);
}

@compute
@workgroup_size(1)
fn madwide1_test() {
  var chain: chain_t;
  var c  = vec2<u32>(v_indices[2], v_indices[3]); 
  
  v_indices[0] = madwide1(&chain, v_indices[0], v_indices[1], c).x;
  v_indices[1] = madwide1(&chain, v_indices[0], v_indices[1], c).y;
  v_indices[2] = u32(chain.firstOperation);
}