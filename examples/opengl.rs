use std::{fs::OpenOptions, io::Read, time::Duration, collections::HashMap};
use objld::*;
use rayon::{iter::{ParallelIterator}};
use std::hash::Hash;

#[derive(Debug, PartialEq, Eq, Hash)]
struct ParsedVertex{
    pos: (F32Wrapper, F32Wrapper, F32Wrapper),
    tex: (F32Wrapper, F32Wrapper),
    norm: (F32Wrapper, F32Wrapper, F32Wrapper)
}

#[derive(Debug, Clone, Copy)]
struct F32Wrapper{
    inner: f32
}
impl F32Wrapper{
    fn key(&self) -> u64{
        let prec: u16 = 7;// digits in b10
        let whole = self.inner as i64;
        let frac = ((self.inner as f64 - whole as f64) * 10.0_f64.powi(prec.into())) as i64;
        (whole*10i64.pow(prec.into()) + frac) as u64
    }
}

impl PartialEq for F32Wrapper{
    fn eq(&self, other: &Self) -> bool {
        self.key() == other.key()
    }
}
impl Eq for F32Wrapper{}
impl Hash for F32Wrapper{
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.key().hash(state)
    }
}

impl From<f32> for F32Wrapper{
    fn from(v: f32) -> Self {
        Self{inner: v}
    }
}

impl From<F32Wrapper> for f32{
    fn from(v: F32Wrapper) -> Self{
        v.inner
    }
}

#[derive(Default)]
struct RawData3D{
    vertex_pos: Vec<(f32, f32, f32)>,
    vertex_tex: Vec<(f32, f32)>,
    vertex_norm: Vec<(f32, f32, f32)>,
    vert_ind: Vec<objld::VertexIndeces<i32>>
}

fn to_raw_data3d(parsed_data: Vec<LineResult<f32, i32>>) -> RawData3D{
    let mut r = RawData3D::default();
    for line in parsed_data{
        match line{
            LineResult::VertDataLine(v) => match v{
                VertexData::Coord3 { x, y, z } => {r.vertex_pos.push((x, y, z));},
                VertexData::Normal { x, y, z } => {r.vertex_norm.push((x, y, z)); },
                VertexData::TextureCoord2 { u, v } => {r.vertex_tex.push((u, v))},
                _ => {},
            },
            LineResult::FaceLine(f) => match f{
                Face::Face3 { v1, v2, v3 } => {r.vert_ind.push(v1); r.vert_ind.push(v2); r.vert_ind.push(v3);},
                Face::Face4 { v1, v2, v3, v4 } => {r.vert_ind.push(v1); r.vert_ind.push(v2); r.vert_ind.push(v3);  r.vert_ind.push(v3); r.vert_ind.push(v4); r.vert_ind.push(v1); },
            },
            LineResult::NoData => {},
            LineResult::Error(_e) => {println!("{}", _e)} // Ignore unparsed data
        }
    }
    r
}

#[derive(Default)]
struct OpenGLData3D{
    vertex_pos: Vec<(f32, f32, f32)>,
    vertex_tex: Vec<(f32, f32)>,
    vertex_norm: Vec<(f32, f32, f32)>,
    indecies: Vec<u32>
}

fn to_opengl_data3d_dedup(raw_data: RawData3D) -> OpenGLData3D{
    let mut o = OpenGLData3D::default();
    let mut h: HashMap<ParsedVertex, u32> = HashMap::new();
    h.reserve(raw_data.vertex_pos.len());
    let mut curr_ind: u32 = 0;
    for v in raw_data.vert_ind{
        let coord_ind = v.coord_rindex;
        let tex_ind = v.texcoord_rindex.unwrap();
        let norm_ind = v.normal_rindex.unwrap();
        let pos_tuple = raw_data.vertex_pos[if coord_ind < 0 { raw_data.vertex_pos.len() as i32 + coord_ind} else {coord_ind} as usize];
        let tex_tuple = raw_data.vertex_tex[if tex_ind < 0 { raw_data.vertex_tex.len() as i32 + tex_ind} else {tex_ind} as usize];
        let norm_tuple = raw_data.vertex_norm[if norm_ind < 0 { raw_data.vertex_norm.len() as i32 + norm_ind} else {norm_ind} as usize];
        let pv = ParsedVertex{
            pos: (pos_tuple.0.into(), pos_tuple.1.into(), pos_tuple.2.into()),
            tex: (tex_tuple.0.into(), tex_tuple.1.into()),
            norm: (norm_tuple.0.into(), norm_tuple.1.into(), norm_tuple.2.into())
          };
          if let Some(repeated_ind) = h.get(&pv){
              o.indecies.push(*repeated_ind);
          }else{
              h.insert(pv, curr_ind);
              o.indecies.push(curr_ind);
              curr_ind += 1;
          }
    }
    o.vertex_pos.resize(h.len(), (0.0, 0.0, 0.0));
    o.vertex_tex.resize(h.len(), (0.0, 0.0));
    o.vertex_norm.resize(h.len(), (0.0, 0.0, 0.0));
    for (v, i) in h {
        o.vertex_pos[i as usize] = (v.pos.0.into(), v.pos.1.into(), v.pos.2.into());
        o.vertex_tex[i as usize] = (v.tex.0.into(), v.tex.1.into());
        o.vertex_norm[i as usize] = (v.norm.0.into(), v.norm.1.into(), v.norm.2.into());
    }
    o
}

fn to_opengl_data3d_simple(raw_data: RawData3D) -> OpenGLData3D{
    let mut o = OpenGLData3D::default();
    o.vertex_pos.reserve(raw_data.vert_ind.len());
    o.vertex_tex.reserve(raw_data.vert_ind.len());
    o.vertex_norm.reserve(raw_data.vert_ind.len());
    o.indecies.reserve(raw_data.vert_ind.len());
    let mut curr_ind = 0;
    for v in raw_data.vert_ind{
        let coord_ind = v.coord_rindex;
        let tex_ind = v.texcoord_rindex.unwrap();
        let norm_ind = v.normal_rindex.unwrap();
        let pos_tuple = raw_data.vertex_pos[if coord_ind < 0 { raw_data.vertex_pos.len() as i32 + coord_ind} else {coord_ind} as usize];
        let tex_tuple = raw_data.vertex_tex[if tex_ind < 0 { raw_data.vertex_tex.len() as i32 + tex_ind} else {tex_ind} as usize];
        let norm_tuple = raw_data.vertex_norm[if norm_ind < 0 { raw_data.vertex_norm.len() as i32 + norm_ind} else {norm_ind} as usize];
        o.vertex_pos.push(pos_tuple);
        o.vertex_tex.push(tex_tuple);
        o.vertex_norm.push(norm_tuple);
        o.indecies.push(curr_ind);
        curr_ind += 1;
    }
    o
}

fn main(){
    let t;
    
    let raw_data: RawData3D = {
        let mut f = OpenOptions::new().read(true).write(false).open("rungholt.obj").expect("Opening .obj file!");
        let mut buf = String::new();
        f.read_to_string(&mut buf).expect("Reading .obj file!");
        t = std::time::Instant::now();
        let parsed = parse_file(&buf).collect();
        println!("Prasing took: {}s!", Duration::as_secs_f32(&(std::time::Instant::now()-t)));
        to_raw_data3d(parsed)
    };
    
    let raw_no_verts = raw_data.vertex_pos.len();
    println!("No. of vertices before: {}", raw_no_verts);
    println!("Marker: {}s!", Duration::as_secs_f32(&(std::time::Instant::now()-t)));
    let o = to_opengl_data3d_dedup(raw_data);
    println!("Transforming to opengl data: {}s", Duration::as_secs_f32(&(std::time::Instant::now()-t)));
    println!("No. of vertices after: {}", o.vertex_pos.len()); // Assumes o.vertex_pos.len() == o.vertex_tex.len() == o.vertex_norm.len()
    println!("Increase: {}% more than raw", ((o.vertex_pos.len() as f64 - raw_no_verts as f64)/raw_no_verts as f64) * 100.0);
    println!("Final parsing time: {}s", Duration::as_secs_f32(&(std::time::Instant::now()-t)));
}