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

    // === Forward Pass (Equations) ===
    // Hidden neuron
    double zh = (x1 * w1h) + (x2 * w2h) + bh;
    double h  = Sigmoid(zh);

    // Output neuron
    double zo      = (h * who) + bo;
    double output  = Sigmoid(zo);

    // === Loss (Mean Squared Error, for example) ===
    double error   = output - target;  // or (output - y)
    double loss    = 0.5 * error * error;

    // === Backpropagation (Chain Rule) ===
    // 1) For the output neuron
    double dLoss_dOut = (output - target);          // ∂Loss/∂output
    double dOut_dZo   = output * (1.0 - output);    // ∂output/∂zo (sigmoid derivative)
    double dLoss_dZo  = dLoss_dOut * dOut_dZo;      // chain them

    // Derivative w.r.t. who
    double dZo_dWho   = h;                          // ∂zo/∂who
    double dLoss_dWho = dLoss_dZo * dZo_dWho;       // ∂Loss/∂who

    // Derivative w.r.t. bo
    double dZo_dBo    = 1.0;                       
    double dLoss_dBo  = dLoss_dZo * dZo_dBo;        // ∂Loss/∂bo

    // 2) For the hidden neuron
    //   We first see how much the hidden neuron contributed to error
    double dLoss_dH   = dLoss_dZo * who;            // ∂Loss/∂h
    double dH_dZh     = h * (1.0 - h);              // ∂h/∂zh (sigmoid derivative)
    double dLoss_dZh  = dLoss_dH * dH_dZh;          // chain them

    // Derivative w.r.t. w1h
    double dZh_dw1h   = x1;                        
    double dLoss_dw1h = dLoss_dZh * dZh_dw1h;       // ∂Loss/∂w1h

    // Derivative w.r.t. w2h
    double dZh_dw2h   = x2;
    double dLoss_dw2h = dLoss_dZh * dZh_dw2h;       // ∂Loss/∂w2h

    // Derivative w.r.t. bh
    double dZh_dBh    = 1.0;
    double dLoss_dBh  = dLoss_dZh * dZh_dBh;        // ∂Loss/∂bh

    // === Weight Updates (Gradient Descent) ===
    w1h = w1h - learningRate * dLoss_dw1h;
    w2h = w2h - learningRate * dLoss_dw2h;
    bh  = bh  - learningRate * dLoss_dBh;
    who = who - learningRate * dLoss_dWho;
    bo  = bo  - learningRate * dLoss_dBo;
    */



    //start of machine learning:
    public static void Main()
    {
        double x1 = 1;
        double x2 = 0;
        double y = 1;

        double w1h = 0.4;
        double w2h = 0.3;
        double who = 0.5;

        double bh = 0.1;
        double bo = 0.0;

        double zh = (x1 * w1h) + (x2 * w2h) + bh;
        double h = Sigmoid(zh);

        Console.WriteLine("Z: " + zh);
        Console.WriteLine("Sigmoid: " + h);
    }
 
    public static double Sigmoid(double x)
    {
        return 1.0 / (1.0 + Math.Exp(-x));
    }

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

