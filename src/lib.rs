use std::str::FromStr;

use nom::{IResult, branch::alt, bytes::complete::tag, character::complete::{digit0, digit1, one_of, space0, space1}, combinator::{eof, opt, rest}, sequence::tuple};
use nom::combinator::{map, recognize};
use nom::character::complete::char;
use rayon::{iter::ParallelIterator, str::ParallelString};


#[derive(Debug, PartialEq)]
pub enum VertexData<T>{
    Coord2{ x: T, y: T }, // Unofficial 
    Coord3{ x: T, y: T, z: T },

    // No support for w in Coords

    Normal{x: T, y: T, z: T},

    TextureCoord3{ u: T, v: T, w: T },
    TextureCoord2{ u: T, v: T },
    TextureCoord1{ u: T }
}


#[derive(Debug, PartialEq, Clone, Copy)]
pub struct VertexIndeces<I>{
    pub coord_rindex: I,
    pub texcoord_rindex: Option<I>,
    pub normal_rindex: Option<I>,
}

#[derive(Debug, PartialEq)]
pub enum Face<I>{
    Face3{
        v1: VertexIndeces<I>,
        v2: VertexIndeces<I>,
        v3: VertexIndeces<I>        
    },

    Face4{
        v1: VertexIndeces<I>,
        v2: VertexIndeces<I>,
        v3: VertexIndeces<I>,
        v4: VertexIndeces<I>
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_float1() {
        let (_, res) = consume_num("-1234.1, hello!").unwrap();
        assert_eq!(res, "-1234.1");
    }

    #[test]
    fn test_float2() {
        let (_, res) : (_, f32) = parse_float("2, hj!").unwrap();
        assert_eq!(res, 2.0);
    }

    #[test]
    fn test_vert(){
        let (unconsumed, res) : (_, VertexData<f32>) = parse_coord2(" v 1.0 -2.0 # hi!").unwrap();
        assert_eq!(res, VertexData::Coord2{x: 1.0, y: -2.0});
        assert_eq!(unconsumed, " # hi!");
    }

    #[test]
    fn test_texvert(){
        let (_, res) : (_, LineResult<f32, u32>) = parse_line("vt 1.0 +2.0 3.0").unwrap();
        if let LineResult::VertDataLine(res) = res{
            assert_eq!(res, VertexData::TextureCoord3{u: 1.0, v: 2.0, w: 3.0});
        }else{ panic!("Wrong line type!"); }
    }

    #[test]
    fn test_vert_space(){
        let (_, res) : (_, LineResult<f32, u32>) = parse_line("    v      -5.000000       5.000000       0.000000").unwrap();
        if let LineResult::VertDataLine(res) = res{
            assert_eq!(res, VertexData::Coord3{x: -5.0, y: 5.0, z: 0.0});
        }else{ panic!("Wrong line type!"); }
    }

    #[test]
    fn test_normal(){
        let res : IResult<_, VertexData<f32>> = parse_normal(" vn 1.0 -2.0 # hi!");
        assert_eq!(res, Err(nom::Err::Error(nom::error::Error::new("# hi!", nom::error::ErrorKind::Digit)))) 
    }
    #[test]
    fn test_num1(){
        let (_, res) : (_, i32) = parse_num("1").unwrap();
        assert_eq!(res, 1i32);
    }

    #[test]
    fn test_face1(){
        use std::convert::TryInto;
        let (_, res) : (_, LineResult<f32, u32>) = parse_line("    f 1/2/3 3//2 2/1/").unwrap();
        if let LineResult::FaceLine(Face::Face3{v1, v2, v3}) = res{
            assert_eq!(v1, VertexIndeces::<u32>{coord_rindex: 1.try_into().unwrap(), texcoord_rindex: Some(2.try_into().unwrap()), normal_rindex: Some(3.try_into().unwrap())});
            assert_eq!(v2, VertexIndeces::<u32>{coord_rindex: 3.try_into().unwrap(), texcoord_rindex: None,    normal_rindex: Some(2.try_into().unwrap())});
            assert_eq!(v3, VertexIndeces::<u32>{coord_rindex: 2.try_into().unwrap(), texcoord_rindex: Some(1.try_into().unwrap()), normal_rindex: None});
        }else{ panic!("Wrong line type!"); }
    }
}

// A line can either contain vertex info or face info as far as this parser is concerned
#[derive(Debug)]
pub enum LineResult<'a, T, I>{
    VertDataLine(VertexData<T>),
    FaceLine(Face<I>),
    NoData,
    Error(nom::Err<nom::error::Error<&'a str>>)
}

// Note: Basically only parallel function
pub fn parse_file<'input, T, I>(input: &'input str) -> impl ParallelIterator<Item = LineResult<T, I>>
where T: Send + FromStr + 'input, I: Send + FromStr + 'input{
    input.par_split('\n')
    .map(|line|
        parse_line(line)
        .map(|(_unconsumed, parsed)| parsed)
        .unwrap_or_else(|e|{
            LineResult::Error(e)
        })
    )
}


pub fn parse_line<T, I>(input: &str) -> IResult<&str, LineResult<T, I>>
where T: FromStr, I: FromStr{
    use LineResult::VertDataLine;
    use LineResult::FaceLine;
    use LineResult::NoData;
    alt((
        map(end_line, |_| NoData), // If the line doesn't contain anything just return None

        // 1 float
        map(tuple((parse_texcoord1, end_line)), |(v, _)| VertDataLine(v) ),

        // 2 floats
        map(tuple((parse_coord2, end_line)), |(v, _)| VertDataLine(v) ),
        map(tuple((parse_texcoord2, end_line)), |(v, _)| VertDataLine(v) ),

        // 3 floats
        map(tuple((parse_coord3, end_line)), |(v, _)| VertDataLine(v) ),
        map(tuple((parse_normal, end_line)), |(v, _)| VertDataLine(v) ),
        map(tuple((parse_texcoord3, end_line)), |(v, _)| VertDataLine(v) ),

        // 3 fields
        map(tuple((parse_face3, end_line)), |(f, _)| FaceLine(f)),

        // 4 fields
        map(tuple((parse_face4, end_line)), |(f, _)| FaceLine(f)),
    ))(input)

}

/// Primitive parsers
/**********************************************************************************/
#[inline]
fn consume_num(input: &str) -> IResult<&str, &str>{ recognize(tuple( ( opt(one_of("+-")), digit1, opt(char('.')), digit0) ) )(input) }

#[inline]
fn parse_float<T>(input: &str) -> IResult<&str, T>
where T: FromStr{
    let (input, num) = consume_num(input)?;
    let val: T = T::from_str(num).map_err(|_| nom::Err::Error(nom::error::Error::new(num, nom::error::ErrorKind::Float)))?;
    return Ok((input, val));
}

#[inline]
fn parse_num<T>(input: &str) -> IResult<&str, T>
where T: FromStr{
    let (input, num) = consume_num(input)?;
    let val: T = str::parse(num).map_err(|_| nom::Err::Error(nom::error::Error::new(num, nom::error::ErrorKind::Digit)))?;
    return Ok((input, val));
}

#[inline]
fn end_line(input: &str) -> IResult<&str, &str>{
    type Comment<'a> = &'a str;
   fn consume_comment(input: &str) -> IResult<&str, Comment> { recognize(tuple((space0, char('#'), rest)))(input) }
   recognize(tuple((  space0, opt(consume_comment), eof  )))(input)
}
/**********************************************************************************/

// For 2d vertex coords
fn parse_coord2<T>(input: &str) -> IResult<&str, VertexData<T>>
where T: FromStr {
    let (input, data) = tuple(( space0, tag("v"), space1, parse_float, space1, parse_float ))(input)?;
    Ok((input, VertexData::Coord2{x: data.3, y: data.5}))
}

// For 3d vertex coords
fn parse_coord3<T>(input: &str) -> IResult<&str, VertexData<T>>
where T: FromStr {
    let (input, data) = tuple(( space0, tag("v"), space1, parse_float, space1, parse_float, space1, parse_float ))(input)?;
    Ok((input, VertexData::Coord3{x: data.3, y: data.5, z: data.7}))
}

// For 3d normals (no support for 2d normals)
fn parse_normal<T>(input: &str) -> IResult<&str, VertexData<T>>
where T: FromStr {
    let (input, data) = tuple(( space0, tag("vn"), space1, parse_float, space1, parse_float, space1, parse_float ))(input)?;
    Ok((input, VertexData::Normal{x: data.3, y: data.5, z: data.7}))
}

// For 1D textures
fn parse_texcoord1<T>(input: &str) -> IResult<&str, VertexData<T>>
where T: FromStr {
    let (input, data) = tuple(( space0, tag("vt"), space1, parse_float ))(input)?;
    Ok((input, VertexData::TextureCoord1{u: data.3}))
}

// For 2D textures
fn parse_texcoord2<T>(input: &str) -> IResult<&str, VertexData<T>>
where T: FromStr {
    let (input, data) = tuple(( space0, tag("vt"), space1, parse_float, space1, parse_float ))(input)?;
    Ok((input, VertexData::TextureCoord2{u: data.3, v: data.5}))
}

// For 3D textures
fn parse_texcoord3<T>(input: &str) -> IResult<&str, VertexData<T>>
where T: FromStr{
    let (input, data) = tuple(( space0, tag("vt"), space1, parse_float, space1, parse_float, space1, parse_float ))(input)?;
    Ok((input, VertexData::TextureCoord3{u: data.3, v: data.5, w: data.7}))
}

// For face3 and face4
fn parse_face_vertex<I>(input: &str) -> IResult<&str, VertexIndeces<I>>
where I: FromStr {
    let (input, data) = tuple(( parse_num, char('/'), opt(parse_num), char('/'), opt(parse_num) ))(input)?; // NUM/OPT(NUM)/OPT(NUM)
    Ok((input, VertexIndeces{  coord_rindex: data.0, texcoord_rindex: data.2, normal_rindex: data.4 }))
}

// For triangle faces
fn parse_face3<I>(input: &str) -> IResult<&str, Face<I>>
where I: FromStr {
    let (input, data) = tuple(( space0, tag("f"), space1, parse_face_vertex, space1, parse_face_vertex, space1, parse_face_vertex ))(input)?;
    Ok((input, Face::Face3{ v1: data.3, v2: data.5, v3: data.7 })) // Intentionally ignore data.9
}

// For square faces
fn parse_face4<I>(input: &str) -> IResult<&str, Face<I>>
where I: FromStr {
    let (input, data) = tuple(( space0, tag("f"), space1, parse_face_vertex, space1, parse_face_vertex, space1, parse_face_vertex, space1, parse_face_vertex ))(input)?;
    Ok((input, Face::Face4{ v1: data.3, v2: data.5, v3: data.7, v4: data.9 }))
}
