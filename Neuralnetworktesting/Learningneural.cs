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

        //target
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
        //json data saving for model
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



    /*
    //two layer neural network
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

