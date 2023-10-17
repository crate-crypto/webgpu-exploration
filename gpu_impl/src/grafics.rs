use plotters::prelude::*;

/// function for building a plot from given points
/// input data:
/// points1, points2 - point for building a plot
/// output data:
/// result of building a plot
pub fn build_plot(points1: &[f32], points2: &[f32]) -> Result<(), Box<dyn std::error::Error>> {
  let root = BitMapBackend::new("column_chart.png", (800, 600)).into_drawing_area();
  root.fill(&WHITE)?;

  let mut chart = ChartBuilder::on(&root)
  .set_label_area_size(LabelAreaPosition::Left, 40)
  .set_label_area_size(LabelAreaPosition::Bottom, 40)
  .caption("CPU vs GPU Column Chart", ("sans-serif", 40))
  .build_cartesian_2d(0..2*points1.len() as i32, 0f32..4f32)?;

  chart
  .configure_mesh()
  .disable_x_mesh()
  .y_desc("Time(s)")
  .x_desc("Batches")
  .y_label_formatter(&|y| format!("{:.0}", y)) 
  .x_label_formatter(&|x| {
    if x % 2 == 0 {
      format!("{}", 10_i32.pow((*x / 2) as u32))
    }
    else {
      format!("{}", 10_i32.pow(((*x + 1) / 2) as u32) / 2)
    }
  })
  .draw()?;

  chart
    .draw_series(
        Histogram::vertical(&chart)
          .style(GREEN.mix(0.5).filled())
          .data(points1.iter().enumerate().map(|(i, &val)| ((2*i) as i32 , val))),
    )?
    .label("cpu time")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], GREEN.filled()));

  chart
    .draw_series(
        Histogram::vertical(&chart)
          .style(BLUE.mix(0.5).filled())
          .data(points2.iter().enumerate().map(|(i, &val)| ((((2*i) + 1) as i32), val))),
    )?
    .label("gpu time")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE.filled()));

  // Draw a red vertical line at X = 1500
  chart
    .draw_series(LineSeries::new(
      vec![(7, 0f32), (7, 4f32)], // Start point and end point of the line
      RED.filled(), // Red color
    )
  .point_size(4))?
  .label("equal time line")
  .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED.filled()));
  
  chart
  .configure_series_labels()
  .background_style(&WHITE.mix(0.8))
  .border_style(&BLACK)
  .position(SeriesLabelPosition::UpperRight)
  .legend_area_size(30)
  .draw()?;

  Ok(())
}

