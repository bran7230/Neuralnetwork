using System;
using System.Diagnostics.Tracing;
using System.Runtime.CompilerServices;
using Newtonsoft.Json;
using System.Linq;

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

    /*
      //Types like chatgpt
       foreach (char c in userinput)
      {  
          Console.Write(c);
          Thread.Sleep(50); 
      }
      */
    public static double Sigmoid(double x)
    {
        return 1.0 / (1.0 + Math.Exp(-x));
    }
    // Softmax function
    static double[] Softmax(double[] values)
    {
        double[] expValues = values.Select(Math.Exp).ToArray();
        double sumExp = expValues.Sum();
        return expValues.Select(x => x / sumExp).ToArray();
    }


    //start of RNN model testing and encoded inputs  with some writing animation.
    /***************************************************************************
     ****************************************************************************/

    //dictionary for encoded word inputs
   static Dictionary<string, int> vocabulary = new Dictionary<string, int>
    {
        { "hello", 0 },
        { "hi", 1 },
        { "goodbye", 2 },
        { "how", 3 },
        { "are", 4 },
        { "you", 5 },
        { "name", 6 },
        { "what's", 7 }
     
    };
    //testing
    static void Main()
    {
        
        Console.WriteLine("Hello");

        //takes userinput, then runs the encode method to get int values, then writes back
        String userinput = Console.ReadLine().ToLower();
        List<int> encoded = EncodeInput(userinput);
        Console.WriteLine("Encoded input: "+string.Join(",", encoded));

        //Types like chatgpt
         foreach (char c in userinput)
        {
            Console.Write(c);

            Thread.Sleep(50); 
        }
        
        
    }
    //makes a list with the vocab, runs the input against it, if it flags, return the encoded values, else,  return -1 for unknown.
    static List<int>  EncodeInput(string input)
    {
        //taking inputs, and splitting to ensure words get taken.
        String[] words = input.Split(' ');
        //new vocab
        List<int> output = new List<int>();
        //goes through each word in the inputs
        foreach (String word in words)
        {
            //if it flags, add word
            if (vocabulary.ContainsKey(word))
            {
                output.Add(vocabulary[word]);
            }
            //else, return -1 and display what word was not in vocab
            else
            {
                Console.WriteLine($"Unknown Word: {word}");
                output.Add(-1);
            }

        }
        //return the final list to main
        return output;

    }
    /*
    //softmax test
    static void Main()
    {
        double learningRate = 0.1;
        //z values from greet, code, help
        double[] logits = { 1.0885, 0.621, 1.242 };
        Console.WriteLine("Input Values (Logits):");
        foreach (var logit in logits)
        {
            Console.WriteLine(logit);
        }

        double[] probabilities = Softmax(logits);

        Console.WriteLine("\nSoftmax Probabilities:");
        foreach (var prob in probabilities)
        {
            Console.WriteLine(prob);
        }

        double loss = -(1 * Math.Log(0.358) + 0 * Math.Log(0.224) + 0 * Math.Log(0.417));

        Console.WriteLine(loss);

        double errorgreet = 0.358 - 1;
        double errorcode = 0.224 - 0;
        double errorhelp = 0.417 - 0;

        Console.WriteLine("Error greet: " + errorgreet);
        Console.WriteLine("Error code: " + errorcode);
        Console.WriteLine("Error help: "+errorhelp);

        double newGreet1 = 0.4 - (learningRate * errorgreet);
        double newGreet2 = 0.3 - (learningRate * errorgreet);
        double newGreet3 = 0.7 - (learningRate * errorgreet);
        Console.WriteLine("Greeting gradient decent: ");
        Console.WriteLine("Neuron 1: " + newGreet1 + "\nNeuron 2: "+newGreet2+"\nNeuron 3: "+newGreet3);

        double newCode1 = 0.5 - (learningRate * errorcode);
        double newCode2 = 0.2 - (learningRate * errorcode);
        double newCode3 = 0.1 - (learningRate * errorcode);

        Console.WriteLine("Code gradient decent: ");
        Console.WriteLine("Neuron 1: " + newCode1 + "\nNeuron 2: " + newCode2 + "\nNeuron 3: " + newCode3);

       double newHelp = 0.9 - (learningRate * errorhelp);
       double newHelp2 = 0.4 -(learningRate * errorhelp);
       double newHelp3 = 0.3 -(learningRate * errorhelp);

        Console.WriteLine("Help gradient decent: ");
        Console.WriteLine("Neuron 1: " + newHelp + "\nNeuron 2: " + newHelp2 + "\nNeuron 3: " + newHelp3);

       double zgreet = (0.785*0.4642)+(0.750*0.3642)+(0.785*0.7642);
       Console.WriteLine(zgreet);

        double zcode = (0.785 * 0.4776) + (0.750 * 0.1776) + (0.785 * 0.0776);
        Console.WriteLine(zcode);

        double zhelp = (0.785 * 0.858) + (0.750 * 0.3583) + (0.785 * 0.2583);
        Console.WriteLine(zhelp);

        double[] vals = { 1.2374, 0.5690, 1.1450 };

        double[] probs = Softmax(vals);
        Console.WriteLine("New Softmax: ");
        foreach(var prob in probs)
        {
            Console.WriteLine(prob);
        }

        Console.WriteLine("Loss new: ");
        double losss = -Math.Log(0.4124);
        Console.WriteLine(losss);
    }
    */


    /*
    Simple Intent Classifier - Neural Network (C#)

    This program simulates a basic neural network with multiple outputs (greet, code, help).
    It performs:
    - Bag-of-words input processing (e.g., "hello", "code", "help")
    - Forward pass with weighted inputs and sigmoid activation
    - Intent confidence calculation for each output neuron
    - Error and delta (gradient signal) computation using backpropagation
    - Weight update for the active input's intent

    Example:
    Input: "hello"
    Output: Increases confidence in the "greet" intent by adjusting its weight.

    This is a foundational model for building an intelligent chatbot or intent recognition system.

    // add a * / below to run it
    */
    /*

    public static void Main()
    {
        //base values for false.
        double greetingInput = 0;
        double codeInput = 0;
        double helpInput = 0;
        List<string> words = new List<string> { "hello", "code", "help", "please", "bye" };
        Console.WriteLine("Hello: ");
        string input = Console.ReadLine().ToLower();

        //input checks
        if (input == null)
        {
            Console.WriteLine("Enter valid..");
            return;
        }

        else if (input == "hello")
        {
            greetingInput = 2;
        }

        else if (input == "code")
        {
            codeInput = 2;
        }

        else if (input == "help")
        {

            helpInput = 2;
        }
        //default weight
        double greetingWeight = 1.0;
        double codeWeight = 1.0;
        double helpWeight = 1.0;

        double bias = 1.0;
        //for new weights
        double learningRate = 0.1;
        
        //z
        double greetz = (greetingInput * greetingWeight) + bias;
        double codez = (codeInput * codeWeight) + bias;
        double helpz = (helpInput * helpWeight) + bias;

        Console.WriteLine("Greeting z: "+greetz);
        Console.WriteLine("Code z: "+codez);
        Console.WriteLine("Help z: "+helpz);

        //confidences
        double greetConfidence = Sigmoid(greetz);
        double codeConfidence = Sigmoid(codez);
        double helpConfidence = Sigmoid(helpz);

        Console.WriteLine("Greet confidence: "+greetConfidence);
        Console.WriteLine("Code confidence: "+codeConfidence);
        Console.WriteLine("Help confidence: "+helpConfidence);

        //if its above, print.
       if(greetConfidence > 0.8)
        {
            Console.WriteLine("\nHello\n");
        }

       else if(codeConfidence > 0.8)
        {
            Console.WriteLine("\n Enter Code: \n ");
        }

        else if(helpConfidence > 0.8)
        {
            Console.WriteLine("\n What do you need help with?\n");
        }

       //error margins
        double greetError = greetConfidence - 1;
        double codeError = codeConfidence - 1;
        double helpError = helpConfidence - 1;

        Console.WriteLine("Greet error: "+greetError);
        Console.WriteLine("Code error: "+codeError);
        Console.WriteLine("Help error: "+helpError);

        //deltas for it
        double greetDelta = greetError * greetConfidence * (1-greetConfidence);
        double codeDelta = codeError * codeConfidence * (1-codeConfidence);
        double helpDelta = helpError * helpConfidence * (1-helpConfidence);

        Console.WriteLine("Greet delta: "+greetDelta);
        Console.WriteLine("Code delta: "+codeDelta);
        Console.WriteLine("Help delta: "+helpDelta);

        //gradients for it
        double gradientGreet = greetDelta * greetingInput;
        double gradientCode = codeDelta * codeInput;
        double gradientHelp = helpDelta * helpInput;

        Console.WriteLine("Gradient Greet: " + gradientGreet);
        Console.WriteLine("Gradient Code: " + gradientCode);
        Console.WriteLine("Gradient Help: " + gradientHelp);

        //final weight after learning
        double newGreetWeight = greetingWeight - (learningRate * gradientGreet);
        double newCodeWeight = codeWeight - (learningRate * gradientCode);
        double newHelpWeight = helpWeight - (learningRate * gradientHelp);

        Console.WriteLine("New Greeting Weight: " + newGreetWeight);
        Console.WriteLine("New Code Weight: " + newCodeWeight);
        Console.WriteLine("New Help Weight: " + newHelpWeight);
    }

    */


    /*
    //backpropagation
    
    public static void Main()
    {
        //word bag
        List<string> words = new List<string> { "hello", "code", "help", "please", "bye" };

        //inputs
        double greetingInput = 0;
        double codeInput = 0;
        double helpInput = 0;

        //weights
        double greetingWeight = 1;
        double codeWeight = 1;
        double helpWeight = 1;
        
        //bias
        double bias = 1;
        
        //target which is 1
        double target = 1;

        //learning rate
        double learningRate = 0.1;

        Console.WriteLine("Hello, How can I help?: ");
        #pragma warning disable CS8602 // Dereference of a possibly null reference.
        string input = Console.ReadLine().ToLower();
        #pragma warning restore CS8602 // Dereference of a possibly null reference.



        if (input == null)
        {
            Console.WriteLine("Please enter something...");
            return;
        }

        else if (input.Contains("hello"))
        {
            greetingInput = 2;
        }

        else if (input.Contains("code"))
        {
            codeInput = 2;
        }

        else if (input.Contains("help")) {
            helpInput = 2; 
        }


        //Z for sigmoid equation
        double greetz = (greetingInput * greetingWeight) + bias;
        double codez = (codeInput * codeWeight) + bias;
        double helpz = (helpInput * helpWeight) + bias;

        double greetConfidence = Sigmoid(greetz);
        double codeConfidence = Sigmoid(codez);
        double helpConfidence = Sigmoid(helpz);


        if (greetConfidence > 0.8)
        {
            Console.WriteLine($"GreetConfidence {greetConfidence}");
            Console.WriteLine("Hello");
        }

        else if (codeConfidence > 0.8)
        {
            Console.WriteLine("Code help");
        }

        else if (helpConfidence > 0.8)
        {
            Console.WriteLine("What do you need help with?");
        }

        else
        {
            Console.WriteLine("Help:" +helpConfidence);
            Console.WriteLine("Greet: " + greetConfidence);
            Console.WriteLine("Code: "+codeConfidence);
            Console.WriteLine("Adjust vals");
        }

        //errors 
        double errorGreet = greetConfidence - target;
        double errorCode = codeConfidence - target;
        double errorHelp = helpConfidence - target;

        //deltas
        double deltaGreet = errorGreet * greetConfidence * (1-greetConfidence);
        double deltaCode = errorCode * codeConfidence * (1-codeConfidence);
        double deltaHelp = errorHelp * helpConfidence * (1-helpConfidence);

        //finally new weights
        greetingWeight = greetingWeight - (learningRate * deltaGreet);
        codeWeight = codeWeight - (learningRate * deltaCode);
        helpWeight = helpWeight - (learningRate * deltaHelp);

        
       
    }
    //basic sigmoid equation
    public static double Sigmoid(double x)
    {
        return 1.0 / (1.0 + Math.Exp(-x));
    }
    */



    /*
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
    */
    /*
    //COOLEST THING IVE DONE EVER DOWN BELOW(Edit, it was not the coolest thing. Newest is, the RNN) 

    
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

