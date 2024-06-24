use polars::prelude::*;
use std::error::Error;
use std::fs::File;
use kdam::tqdm;

fn main() -> Result<(), Box<dyn Error>> {
    // Read the input Parquet file into a DataFrame
    let input_file = File::open("data/ndlbib.parquet")?;
    let mut df = ParquetReader::new(input_file).finish()?;

    // Use Kakasi for conversion (assuming you have a kakasi::convert function)
    let converted_readings: Vec<String> = tqdm!(df.column("text")?.str()?.iter())
        .map(|text| {
            text.map(|text| {
                let converted_reading = kakasi::convert(text);
                converted_reading.hiragana
            })
        })
        .collect::<Option<Vec<String>>>()
        .ok_or("Error during Kakasi conversion")?;

    // Add the converted readings as a new column to the DataFrame
    df.with_column(Series::new("kakasi_reading", converted_readings))?;

    // Write the DataFrame to a new Parquet file
    let output_file = File::create("data/ndlbib_kakasi.parquet")?;
    ParquetWriter::new(output_file).finish(&mut df)?;

    Ok(())
}
