using System;


class Learningneural
{


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


    public static void Main()
    {
        double x = 2;

        double w1 = 1.0;     // weight to hidden neuron 1
        double w2 = -1.0;     // weight to hidden neuron 2

        double h1 = x * w1;
        double h2 = x * w2;


        double w3 = 0.5;  // hidden1 → output
        double w4 = 0.5; // hidden2 → output

        double output = (h1 * w3) + (h2 * w4);
        Console.WriteLine(output);
    }
}

