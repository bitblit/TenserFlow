import {Logger} from "@bitblit/ratchet/dist/common/logger";
import * as fs from 'fs';
import * as path from 'path';
import * as tf from '@tensorflow/tfjs';

export class CarDataTrainer {
  constructor() {}

  public async run(): Promise<boolean> {
    Logger.info('Running trainer');
    const cars: Car[] = this.readData();
    const clean: CarStripped[] = cars.map(c => {return {mpg: c.Miles_per_Gallon, horsepower: c.Horsepower};})
        .filter(c => !!c.mpg && !!c.horsepower);
    Logger.info('Found %d cars %d clean', cars.length, clean.length);
    // const pts: Point[] = clean.map(c => {return {x:c.horsepower, y: c.mpg}});
    const model: tf.Model = tf.sequential();
    // Input layer
    model.add(tf.layers.dense({inputShape: [1], units: 1, useBias: true}));

    // Output layer
    model.add(tf.layers.dense({units: 1, useBias: true}));

// Convert the data to a form we can use for training.
    const tensorData = convertToTensor(data);
    const {inputs, labels} = tensorData;

// Train the model
    await trainModel(model, inputs, labels);
    console.log('Done Training');

    return true;
  }

  public convertToTensor(data: CarStripped[]): any {
    return tf.tidy(() => {
      // Step 1, shuffle
      tf.util.shuffle(data);

      // Step 2, convert to tensor
      const inputs: number[] = data.map(d => d.horsepower);
      const labels: number[] = data.map(d => d.mpg);

      const inputTensor: Tensor<Rank.R2> = tf.tensor2d(inputs, [inputs.length, 1]);
      const labelTensor: Tensor<Rank.R2> = tf.tensor2d(inputs, [labels.length, 1]);

      // Step 3 : Normalize to 0-1 using min/max scaling
      const inputMax: number = inputTensor.max();
      const inputMin: number = inputTensor.min();
      const labelMax: number = labelTensor.max();
      const labelMin: number = labelTensor.min();

      const normalizedInputs: number[] = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
      const normalizedLabels: number[] = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));

      return {
        inputs: normalizedInputs,
        labels: normalizedLabels,
        // Return min/max for later use
        inputMax,
        inputMin,
        labelMax,
        labelMin
      };
    });
  }

  public readData(): Car[] {
    const data: string = fs.readFileSync(path.join(__dirname, '../static/data/carsData.json')).toString();
    const rval: Car[] = JSON.parse(data);
    return rval;
  }
}

async function trainModel(model, inputs, labels) {
  // Prepare for training
  model.compile({
    optimizer: tf.train.adam(),
    loss: tf.losses.meanSquaredError,
    metrics: ['mse']
  });

  const batchSize: number = 32;
  const epochs: number = 50;

  return await model.fit(inputs, labels, {
    batchsize,
    epochs,
    shuffle: true,
    callbacks: tfvis.show.fitCallbacks(
        {
          name: 'Train performance'
        },
        [
          'loss', 'mse'
        ],
        {
          height: 200, callbacks['onEpochEnd']
        }
    )
  })
}


export interface Car {
  Name: string;
  Miles_per_Gallon: number;
  Cylinders: number;
  Displacement: number;
  Horsepower: number;
  Weight_in_lbs: number;
  Acceleration: number;
  Year: string;
  Origin: string;
}


export interface CarStripped {
  mpg: number;
  horsepower: number;
}

export interface Point {
  x: number;
  y: number;
}