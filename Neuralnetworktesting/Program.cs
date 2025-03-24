using System;
class NeuralNetwork
{




    private static int bias = 1;

    private static double weight1 = 1.2;

    private static double weight2 = 1.0;

    private static double input1 = 10;

    private static double input2 = 5;






    public static void Main()
    {    
        
         double z = ((input1 * weight1) + (input2 * weight2) + bias);
         double sigmoid = (1 / (1 + (Math.Exp(-z))));

         Console.WriteLine("Z is: " + z.ToString());
         Console.WriteLine("Percent of 'Being sure': " + sigmoid.ToString());
    }


}