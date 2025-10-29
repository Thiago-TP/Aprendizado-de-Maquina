// Path-like variables
use std::error::Error;
use std::fs::File;
use std::path::Path;

// Data structures
use nalgebra::{DMatrix, Dim, Matrix, Storage};

pub fn load_csv(path: impl AsRef<Path>, has_headers: bool) -> DMatrix<f64> {
    // Reads (serializes) the CSV
    let mut records = match csv::ReaderBuilder::new()
        .has_headers(has_headers)
        .from_path(path)
    {
        Ok(records) => records,
        _ => panic!("File reading failed!"),
    };

    // Loads (deserializes) the CSV
    let loaded_data = match records
        .deserialize::<Vec<f64>>()
        .collect::<Result<Vec<Vec<f64>>, _>>()
    {
        Ok(loaded_data) => loaded_data,
        _ => panic!("Data loading failed!"),
    };

    // Puts data in a matrix
    let rows = loaded_data.len();
    let cols = loaded_data[0].len();
    let stream: Vec<f64> = loaded_data.into_iter().flatten().collect();

    DMatrix::from_row_slice(rows, cols, &stream)
}

pub fn to_csv<R, C, S>(path: &str, data: &Matrix<f64, R, C, S>) -> Result<(), Box<dyn Error>>
where
    R: Dim,
    C: Dim,
    S: Storage<f64, R, C>,
{
    // Create a new CSV writer
    let file = File::create(path)?;
    let mut wtr = csv::WriterBuilder::new().from_writer(file);

    // Write the records to the CSV file
    for row in data.row_iter() {
        wtr.serialize(row.iter().collect::<Vec<_>>())?;
    }

    // Flush the writer to ensure all data is written
    wtr.flush()?;

    println!("\tData successfully written to {}", path);
    Ok(())
}
