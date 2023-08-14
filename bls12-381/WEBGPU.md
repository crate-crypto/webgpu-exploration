# WGSL (Webgpu shading language)

## Types

* Concrete Scalars
    * `i32`
    * `u32`
    * `f32`
    * `bool`

* Abstract numerics
    * `abstract-float` [64bit]
    * `abstract-int` [64bit]

```rust

const pi = 3.14159265359; // 'pi' is of type abstract-float
const two = 2;            // 'two' is of type abstract-int

// 'two_pi' is an abstract-float
// 'two' was implicitly converted from an abstract-int to abstract-float for the
// multiplication, which is performed with 64-bit floating point precision.
const two_pi = pi * two;

// Abstract-ints can implicitly convert to i32, u32, f32
const implicitly_convert_abstract_int_to_i32 : i32 = 100;
const implicitly_convert_abstract_int_to_u32 : u32 = 100;
const implicitly_convert_abstract_int_to_f32 : f32 = 100;

```


* vecNf -> 2, 3, 4
```
vec2f	is an alias to vec2<f32>.
vec3u	is an alias to vec3<u32>.
vec4i	is an alias to vec4<i32>.
```

* wgsl supports 2x2 and 4x4  matrices
    * the notation is opposite of what is taught in mathematics class.


## entry points 

* `@compute` , `@fragment` , `@vertex`
* 
