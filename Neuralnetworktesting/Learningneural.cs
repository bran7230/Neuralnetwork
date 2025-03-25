using System;
using Newtonsoft.Json;

public class Learningneural
{
    //z = (x1 * w1) + (x2 * w2) + b
    //o(z) or confidence = 1/ 1 + (e^-z)
    // prediction = x * w
    //loss = Math.Pow(prediction - y, 2)
    //error = prediction - target
    //Gradient = 2 * hidden * (output - y)
    /*// === Activation Function ===
    double Sigmoid(double x)
    {
        return 1.0 / (1.0 + Math.Exp(-x));
    }

    // === Forward Pass ===

    // Hidden neuron
    double hiddenInput = (input1 * weightInput1ToHidden) + (input2 * weightInput2ToHidden) + hiddenBias;
    double hiddenOutput = Sigmoid(hiddenInput);

    // Output neuron
    double outputInput = (hiddenOutput * weightHiddenToOutput) + outputBias;
    double predictedOutput = Sigmoid(outputInput);

    // === Loss ===
    double error = predictedOutput - targetOutput;
    double loss = 0.5 * error * error;

    // === Backpropagation ===
    // Output neuron derivatives
    double dError_dPredicted = predictedOutput - targetOutput;
    double dPredicted_dOutputInput = predictedOutput * (1 - predictedOutput);
    double dLoss_dOutputInput = dError_dPredicted * dPredicted_dOutputInput;

    double dLoss_dWeightHiddenToOutput = dLoss_dOutputInput * hiddenOutput;
    double dLoss_dOutputBias = dLoss_dOutputInput * 1;

    // Hidden neuron derivatives
    double dLoss_dHiddenOutput = dLoss_dOutputInput * weightHiddenToOutput;
    double dHiddenOutput_dHiddenInput = hiddenOutput * (1 - hiddenOutput);
    double dLoss_dHiddenInput = dLoss_dHiddenOutput * dHiddenOutput_dHiddenInput;

    double dLoss_dWeightInput1ToHidden = dLoss_dHiddenInput * input1;
    double dLoss_dWeightInput2ToHidden = dLoss_dHiddenInput * input2;
    double dLoss_dHiddenBias = dLoss_dHiddenInput * 1;

    // === Weight Updates ===
    weightInput1ToHidden -= learningRate * dLoss_dWeightInput1ToHidden;
    weightInput2ToHidden -= learningRate * dLoss_dWeightInput2ToHidden;
    hiddenBias           -= learningRate * dLoss_dHiddenBias;
    weightHiddenToOutput -= learningRate * dLoss_dWeightHiddenToOutput;
    outputBias           -= learningRate * dLoss_dOutputBias;

    */



    //start of machine learning:
    public static void Main()
    {
        double x1 = 1;
        double x2 = 0;
        double y = 1;

        double w1h = 0.4; //weight input 1 hidden
        double w2h = 0.3;//weight input 2 hidden
        double who = 0.5; //hidden output weight

        double bh = 0.1; // hidden bias
        double bo = 0.0; // output bias

        double hiddenInput = (x1 * w1h) + (x2 * w2h) + bh;
        double hiddenOutput = Sigmoid(hiddenInput);

        double outputInput = (hiddenOutput * who) + bo; 
        
        double prediction = Sigmoid(outputInput);

        double error = prediction - y;

        double deltaOut = error * (prediction * (1 - prediction));

        double learningRate = 0.1;

   
        
        Console.WriteLine("Z: " + hiddenInput);
        Console.WriteLine("Output input: " + outputInput);
        Console.WriteLine("Prediction: "+prediction);
        Console.WriteLine("Error: "+error);
        Console.WriteLine("Delta: " + deltaOut);

        double gradient_who = deltaOut * hiddenOutput;  // (gradient) for who

        who = who - (learningRate * gradient_who); 
        Console.WriteLine("New Who: " + who);
    }
 
    public static double Sigmoid(double x)
    {
        return 1.0 / (1.0 + Math.Exp(-x));
    }


    //COOLEST SHIT IVE DONE EVER DOWN BELOW
    /*
    //Start of confidence on text

    //making it save now
    public static void Main()
    {
        //inputs
        double x1 = 2;
        double x2 = 3;

        //weights
        double w1 = 0.6;
        double w2 = 0.7;


        double bias = -0.5;

        double learningRate = 0.1;

        //target output
        double y = 1;


        for (int i = 0; i < 10000; i++)
        {
            //calc for z
            double z = (x1 * w1) + (x2 * w2) + bias;
            //confidence
            double confidence = 1 / (1 + Math.Exp(-z));

            double error = confidence - y;

            //gradient for new weights
            double gradientW1 = 2 * x1 * (confidence - y);
            double gradientW2 = 2 * x2 * (confidence - y);
            //new weight set
            w1 = w1- (learningRate * gradientW1);
            w2 = w2- (learningRate * gradientW2);

            Console.WriteLine($"Number: {i}: w1 = {w1:F4}, w2 = {w2:F4}, Confidence = {confidence:F4}, Error = {error:F4}");

        }

        //json data to save for my model
        var jsondata = new
        {

            Input1 = x1,
            Input2 = x2,
            W1 = $"{w1:F4}",
            W2 = $"{w2:F4}",
            bias = bias ,
            learningRate = learningRate ,
            y = y ,

        };
    

        //new weights
        Console.WriteLine($"New weight W1: {w1:F4} ");
        Console.WriteLine($"New weight W2: {w2:F4}");  

        //making it a json file
        string json = JsonConvert.SerializeObject(jsondata, Formatting.Indented);

        string filePath = "C:\\Users\\brand\\OneDrive\\Desktop\\Neuralnetworktesting\\Neuralnetworktesting\\Firstmodel.json";

        using (var stream = new FileStream(filePath, FileMode.OpenOrCreate, FileAccess.Write, FileShare.ReadWrite))

        using (var writer = new StreamWriter(stream))
        {
            writer.Write(json);
        }
    }

    */


   //two layer neural network
    /*
    public static void Main()
    {
        double x = 3;//input value

        double w1 = 2;     // weight to hidden neuron 1
        double w2 = 2;     // weight to hidden neuron 2

        double learningrate = 0.1;

        double h1 = x * w1; //hidden neuron 1
        double h2 = x * w2; //hidden neuron 2


        double w3 = 0.5;  // hidden1 → output
        double w4 = -0.4; // hidden2 → output
        double y = 1; //y val
        double output = (h1 * w3) + (h2 * w4);//output for neruons

        double w3gradient = 2 * h1 * (output - y);

        double neww3 = w3 - (learningrate * w3gradient); //updated weight for neuron 1 or w3

        double w4gradient = 2 * h2 * (output - y); //w4 gradient 

        double neww4 = w4 - (learningrate * w4gradient);//new w4 weight


        Console.WriteLine("Output: " + output);
        Console.WriteLine("w3 Gradient: " + w3gradient);
        Console.WriteLine("New w3: " + neww3);
        Console.WriteLine("w4 Gradient: " + w4gradient);
        Console.WriteLine("New w4: " + neww4);

    }

    

    */

    //simple neurons
    /*
        public static void Main()
        {
            double x = 5;
            double w = 0.6;
            double y = 0;
            double learningRate = 0.1;


            double prediction = x * w;

            double gradient = 2 * x *(prediction - y);

            double newWeight = w - (learningRate * gradient);
            Console.WriteLine("Prediction: " + prediction);
            Console.WriteLine("Gradient: " + gradient);
            Console.WriteLine("New Weight: " + newWeight);
        }
    */


    //examples
    /*
    public static void Main()
    {
        double x = 3;
        double y = 1;
        double w = 0.05;
        double learningRate = 0.05;
     for(int i = 0; i < 10; i++)
        {
            double prediction = x * w;
            Console.WriteLine("Prediction: " + prediction);
            w = Train(x, y, w, learningRate);
            Console.WriteLine("Step" + i +":" + w);
            double loss = Math.Pow(prediction - y, 2);
            Console.WriteLine("Loss: " + loss);

        }
    }

    public static double Train(double x, double y, double w, double learningRate)
    {

        double prediction = x * w;

        double gradient = 2 *x *(prediction - y);

        double newweight = w - (learningRate * gradient);

        return newweight;


    }
    */


}

