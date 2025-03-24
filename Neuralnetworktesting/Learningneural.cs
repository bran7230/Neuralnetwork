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



    public static void Main()
    {
        double x = 5;
        double w = -2.4;
        double y = 0;
        double n = 0.1;


        double prediction = x * w;

        double gradient = 2 * x *(prediction - y);

        double newWeight = w - (n * gradient);

        Console.WriteLine(prediction);
        Console.WriteLine(gradient);
        Console.WriteLine(newWeight);
    }

}

