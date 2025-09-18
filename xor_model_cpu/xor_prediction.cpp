#include <iostream>
#include <cmath>

// Use double for floating-point calculations
using dtype = double;

// Global parameters
const int NUM_INPUTS = 2;
const int NUM_SAMPLES = 4;
const int NUM_EPOCHS = 10000;
const dtype LEARNING_RATE = 0.1;

// Sigmoid activation function
dtype sigmoid(dtype x)
{
    return 1.0 / (1.0 + exp(-x));
}


// Function to train a single gate model with early stopping
void train_gate(dtype *gate_outputs, dtype &w1, dtype &w2, dtype &b, dtype **inputs)
{
    w1 = 0.0;
    w2 = 0.0;
    b = 0.0;

    const int EARLY_STOPPING_EPOCHS = 1000; // Stop if loss doesn't decrease for this many epochs
    dtype previous_loss = 1e18;             // Initialize with a very large value
    int no_improvement_epochs = 0;

    for (int epoch = 0; epoch < NUM_EPOCHS; ++epoch)
    {
        dtype total_error_sum = 0.0;

        for (int i = 0; i < NUM_SAMPLES; ++i)
        {
            dtype z = (inputs[i][0] * w1) + (inputs[i][1] * w2) + b;
            dtype prediction = sigmoid(z);

            dtype error = gate_outputs[i] - prediction;
            total_error_sum += error * error;

            dtype d_z = error * prediction * (1.0 - prediction);

            w1 = w1 + (LEARNING_RATE * d_z * inputs[i][0]);
            w2 = w2 + (LEARNING_RATE * d_z * inputs[i][1]);
            b = b + (LEARNING_RATE * d_z);
        }

        dtype average_loss = total_error_sum / NUM_SAMPLES;
        if (epoch % 1000 == 0)
        {
            std::cout << "Epoch " << epoch << ", Average Loss: " << average_loss << std::endl;
        }

        // Early stopping check
        if (average_loss < previous_loss)
        {
            previous_loss = average_loss;
            no_improvement_epochs = 0; // Reset counter
        }
        else
        {
            no_improvement_epochs++;
        }

        if (no_improvement_epochs >= EARLY_STOPPING_EPOCHS)
        {
            std::cout << "Early stopping at epoch " << epoch << " due to no improvement." << std::endl;
            break;
        }
    }
}

// Function to predict a single gate's output
dtype predict_gate(dtype x1, dtype x2, dtype w1, dtype w2, dtype b)
{
    dtype z = (x1 * w1) + (x2 * w2) + b;
    dtype prediction_prob = sigmoid(z);
    return (prediction_prob >= 0.5) ? 1.0 : 0.0;
}

// Function to perform the full XOR prediction
dtype predict_xor(dtype x1, dtype x2,
                  dtype w1_nand, dtype w2_nand, dtype b_nand,
                  dtype w1_or, dtype w2_or, dtype b_or,
                  dtype w1_and, dtype w2_and, dtype b_and)
{

    // First layer: NAND and OR models
    dtype nand_out = predict_gate(x1, x2, w1_nand, w2_nand, b_nand);
    dtype or_out = predict_gate(x1, x2, w1_or, w2_or, b_or);

    // Second layer: AND model receives outputs from NAND and OR
    dtype xor_prediction = predict_gate(nand_out, or_out, w1_and, w2_and, b_and);
    return xor_prediction;
}

int main()
{
    // Dynamically allocate memory for XOR truth table
    dtype **xor_inputs = new dtype *[NUM_SAMPLES];
    for (int i = 0; i < NUM_SAMPLES; ++i)
    {
        xor_inputs[i] = new dtype[NUM_INPUTS];
    }
    dtype *xor_outputs = new dtype[NUM_SAMPLES];

    // Populate XOR data
    xor_inputs[0][0] = 0;
    xor_inputs[0][1] = 0;
    xor_outputs[0] = 0;
    xor_inputs[1][0] = 0;
    xor_inputs[1][1] = 1;
    xor_outputs[1] = 1;
    xor_inputs[2][0] = 1;
    xor_inputs[2][1] = 0;
    xor_outputs[2] = 1;
    xor_inputs[3][0] = 1;
    xor_inputs[3][1] = 1;
    xor_outputs[3] = 0;

    // Dynamically allocate memory for intermediate gate outputs
    dtype *nand_outputs = new dtype[NUM_SAMPLES];
    nand_outputs[0] = 1;
    nand_outputs[1] = 1;
    nand_outputs[2] = 1;
    nand_outputs[3] = 0;

    dtype *or_outputs = new dtype[NUM_SAMPLES];
    or_outputs[0] = 0;
    or_outputs[1] = 1;
    or_outputs[2] = 1;
    or_outputs[3] = 1;

    dtype *and_outputs = new dtype[NUM_SAMPLES];
    and_outputs[0] = 0;
    and_outputs[1] = 0;
    and_outputs[2] = 0;
    and_outputs[3] = 1;

    // Parameters for each model
    dtype w1_nand, w2_nand, b_nand;
    dtype w1_or, w2_or, b_or;
    dtype w1_and, w2_and, b_and;

    std::cout << "Starting training for NAND, OR, and AND models..." << std::endl;

    // Train the individual models
    train_gate(nand_outputs, w1_nand, w2_nand, b_nand, xor_inputs);
    train_gate(or_outputs, w1_or, w2_or, b_or, xor_inputs);
    train_gate(and_outputs, w1_and, w2_and, b_and, xor_inputs);

    std::cout << "Training finished. Verifying XOR predictions:" << std::endl;
    for (int i = 0; i < NUM_SAMPLES; ++i)
    {
        dtype x1 = xor_inputs[i][0];
        dtype x2 = xor_inputs[i][1];

        dtype predicted_xor = predict_xor(x1, x2,
                                          w1_nand, w2_nand, b_nand,
                                          w1_or, w2_or, b_or,
                                          w1_and, w2_and, b_and);

        std::cout << "Input [" << x1 << ", " << x2 << "], Expected XOR: " << xor_outputs[i]
                  << ", Predicted XOR: " << predicted_xor << std::endl;
    }

    // Clean up dynamically allocated memory
    for (int i = 0; i < NUM_SAMPLES; ++i)
    {
        delete[] xor_inputs[i];
    }
    delete[] xor_inputs;
    delete[] xor_outputs;
    delete[] nand_outputs;
    delete[] or_outputs;
    delete[] and_outputs;

    return 0;
}
