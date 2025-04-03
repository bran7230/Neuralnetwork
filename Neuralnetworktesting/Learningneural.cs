using System.Reflection;
using System.Xml.Schema;

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
  

public class Tokenizer
    {
        private Dictionary<string, int> wordToId = new Dictionary<string, int>();
        private Dictionary<int, string> idToWord = new Dictionary<int, string>();
        private const int START_TOKEN = 0;
        private const int END_TOKEN = 1;

        public Tokenizer()
        {
            // Initialize vocabularies
            wordToId["<START>"] = START_TOKEN;
            wordToId["<END>"] = END_TOKEN;
            idToWord[START_TOKEN] = "<START>";
            idToWord[END_TOKEN] = "<END>";
            wordToId["I"] = 2; idToWord[2] = "I";
            wordToId["am"] = 3; idToWord[3] = "am";
            wordToId["Ada"] = 4; idToWord[4] = "Ada";
        }

        public List<int> Encode(string scentence)
        {
            var tokens = new List<int> { START_TOKEN };
            foreach (var word in scentence.Split(" "))
            {
                if (wordToId.ContainsKey(word)){
                    tokens.Add(wordToId[word]);
                }
            }

            tokens.Add(END_TOKEN);
            return tokens;
        }

        public string Decode(List<int> tokenIds)
        {
            var words = new List<string>();
            foreach (var id in tokenIds)
            {
                if (idToWord.ContainsKey(id))
                {
                    words.Add(idToWord[id]);
                }
            }
            return string.Join(" ", words);
        }
    }

    public static void Main()
    {
      




    }

    /*
    //two layer transformer, actually learns
    
    // Global vocab dictionaries
    static Dictionary<int, string> idToWord = new Dictionary<int, string>
{
    { 0, "hello" },
    { 1, "world" },
    { 2, "I" },
    { 3, "am" },
    { 4, "Test" }
};

    static Dictionary<string, int> wordToId = idToWord.ToDictionary(kv => kv.Value, kv => kv.Key);

    public static void Main()
    {
        int epochs = 1000;

      
        TransformerModel model = new TransformerModel(numLayers: 2);

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            string inputWord = "I";
            string targetWord = "am";

            int inputId = wordToId[inputWord];
            int targetId = wordToId[targetWord];

            double[] inputVector = new double[] { 0, 0, 1, 0, 0 };
            double[] logits = model.Forward(inputVector);


            double[] probs = Softmax(logits);

            int predictedId = Array.IndexOf(probs, probs.Max());
            string predictedWord = idToWord[predictedId];

           

            double loss = -Math.Log(probs[targetId]);
            if (epoch % 100 == 0)
            {
                Console.WriteLine($"Epoch {epoch} | Loss: {loss:F4} | Predicted: {predictedWord}");
            }

            double[] delta = new double[probs.Length];
           

            for (int i = 0; i < probs.Length; i++)
            {
                delta[i] = probs[i];
            }
            delta[targetId] -= 1.0;
            model.UpdateOutputWeights(delta, 0.1);
        }


    }
    
    //N Layer Transformer
    public class TransformerBlock
    {
        private double[,] W1;
        private double[] b1;
        private double[,] W2;
        private double b2;
        private double[] lastZ1;

        public TransformerBlock(double[,] w1, double[] b1, double[,] w2, double b2)
        {
            this.W1 = w1;
            this.b1 = b1;
            this.W2 = w2;
            this.b2 = b2;
        }

        public double[] Forward(double[] input)
        {
            double[] z1 = new double[2];
            z1[0] = (input[0] * W1[0, 0]) + (input[1] * W1[1, 0]) + b1[0];
            z1[1] = (input[0] * W1[0, 1]) + (input[1] * W1[1, 1]) + b1[1];
            lastZ1 = z1;

            for (int i = 0; i < 2; i++)
            {
                if (z1[i] < 0) z1[i] = 0;
            }

            double[] logits = new double[5]; // 5 = vocab size
            for (int i = 0; i < 5; i++)
            {
                logits[i] = (z1[0] * W2[0, i]) + (z1[1] * W2[1, i]) + b2;
            }

            return logits; // no normalization here for now
        }

        public double[] GetLastZ1()
        {
            return lastZ1;
        }


        public void UpdateOutputWeights(double[] delta, double learningRate, double[] z1)
        {
            for (int i = 0; i < W2.GetLength(0); i++) // each hidden neuron
            {
                for (int j = 0; j < W2.GetLength(1); j++) // each output logit
                {
                    W2[i, j] -= learningRate * delta[j] * z1[i];
                }
            }

            b2 -= learningRate * delta.Sum(); // update shared bias
        }


    }
    public class TransformerModel
    {
        private List<TransformerBlock> blocks = new List<TransformerBlock>();

        public TransformerModel(int numLayers)
        {
            for (int i = 0; i < numLayers; i++)
            {

                double[,] W1 = { { 0.5, 0.1 }, { 0.4, 0.6 } };
                double[] b1 = { 0.2, 0.2 };
                double[,] W2 = {
                         { 0.3, 0.4, 0.1, 0.5, 0.2 },
                         { 0.5, 0.6, 0.3, 0.1, 0.4 }
                         }; // shape: 2 x 5

                double b2 = 0.1;

                blocks.Add(new TransformerBlock(W1, b1, W2, b2));
            }
        }

        public double[] Forward(double[] input)
        {
            double[] output = input;

            foreach (var block in blocks)
            {
                output = block.Forward(output);
            }
            return output;
        }
        public void UpdateOutputWeights(double[] delta, double learningRate)
        {
            blocks.Last().UpdateOutputWeights(delta, learningRate, blocks.Last().GetLastZ1());
        }

    }

    */


    /*
    //FINALLY STARTING TRANSFOMRERS!!!!

    static double[] TransformerBlock1(double[] input)
    {
        double[,] W1 = { { 0.5, 0.1 }, { 0.4, 0.6 } };
        double[] b1 = { 0.2, 0.2 };

        double[,] W2 = { { 0.3 }, { 0.5 } };
        double b2 = 0.1;

        double[] z1 = new double[2];

        z1[0] = (input[0] * W1[0, 0]) + (input[1] * W1[1, 0]) + b1[0];
        z1[1] = (input[0] * W1[0, 1]) + (input[1] * W1[1, 1]) + b1[1];

        //applying Relu to not let negatives pass (Relu is a Max(0,xi))
        for (int i = 0; i < z1.Length; i++)
        {
            if (z1[i] < 0)
            {
                z1[i] = 0;
            }
        }

        double ffnOutput = (z1[0] * W2[0, 0]) + (z1[1] * W2[1, 0]) + b2;

        //Hardcoding for tests
        double[] ffnOutVec = { 0.9, 1.4 };

        //Residual connections 
        double[] residual = new double[2];

        for (int i = 0; i < 2; i++)
        {
            residual[i] = ffnOutVec[i] + input[i];
        }

        //Now Normalize values

        double[] normalized = new double[2];

        double mean = (residual[0] + residual[1]) / 2.0;

        //Variance 
        double variance = ((Math.Pow(residual[0] - mean, 2) + Math.Pow(residual[1] - mean, 2)) / 2.0);
        double epsilon = 1e-5;

        //Normalize values

        for (int i = 0; i < 2; i++)
        {
            normalized[i] = (residual[i] - mean) / Math.Sqrt(variance + epsilon);
        }

        //finally return values
        return normalized;
    }

    static double[] TransformerBlock2(double[] input)
    {
        double[,] W1 = {
            { 0.6, 0.2 },
            { 0.3, 0.7 }
            };
        double[] b1 = { 0.1, 0.1 };

        double[,] W2 = {
             { 0.4 },
             { 0.6 }
             };
        double b2 = 0.2;


        double[] z1 = new double[2];

        z1[0] = (input[0] * W1[0, 0]) + (input[1] * W1[1, 0]) + b1[0];
        z1[1] = (input[0] * W1[0, 1]) + (input[1] * W1[1, 1]) + b1[1];

        //applying Relu to not let negatives pass (Relu is a Max(0,xi))
        for (int i = 0; i < z1.Length; i++)
        {
            if (z1[i] < 0)
            {
                z1[i] = 0;
            }
        }

        double ffnOutput = (z1[0] * W2[0, 0]) + (z1[1] * W2[1, 0]) + b2;

        //Hardcoding for tests
        double[] ffnOutVec = { 0.9, 1.4 };

        //Residual connections 
        double[] residual = new double[2];

        for (int i = 0; i < 2; i++)
        {
            residual[i] = ffnOutVec[i] + input[i];
        }

        //Now Normalize values

        double[] normalized = new double[2];

        double mean = (residual[0] + residual[1]) / 2.0;

        //Variance 
        double variance = ((Math.Pow(residual[0] - mean, 2) + Math.Pow(residual[1] - mean, 2)) / 2.0);
        double epsilon = 1e-5;

        //Normalize values

        for (int i = 0; i < 2; i++)
        {
            normalized[i] = (residual[i] - mean) / Math.Sqrt(variance + epsilon);
        }

        //finally return values
        return normalized;
    }


    public static void Main()
    {
        double[] input = { 1.0, 2.0 };

        // Pass through Block 1
        double[] layer1Output = TransformerBlock1(input);

        // Then pass result into Block 2
        double[] finalOutput = TransformerBlock2(layer1Output);

        // Print final result
        Console.WriteLine("Final Transformer Output:");
        for (int i = 0; i < finalOutput.Length; i++)
        {
            Console.WriteLine($"finalOutput[{i}] = {finalOutput[i]}");
        }
    }

    /*
     * First Transformers block
    public static void Main()
    {
        //Matrixes
        double[] input = { 2.0, 1.0 };
        double[,] W1 = {
            { 0.5, 0.1 },
            { 0.4, 0.6 },
        };
        double[,] W2 = {
            { 0.3 },
            { 0.5 }
        };

        double[] b1 = { 0.2, 0.2 }; 
        double b2 = 0.1;
        double[] z1 = new double[2];

        z1[0] =  (input[0] * W1[0, 0]) + (input[1] * W1[1, 0]) + b1[0];
        z1[1] = (input[0] * W1[0,1]) + (input[1] * W1[1,1]) + b1[1];

        //applying Relu to not let negatives pass (Relu is a Max(0,xi))
        for (int i = 0; i < z1.Length; i++)
        {
            if (z1[i] < 0)
            {
                z1[i] = 0;
            }
        }

        double output = (z1[0] * W2[0, 0]) + (z1[1] * W2[1, 0]) + b2;
        Console.WriteLine(output);

        //Residual Connections 
        double[] ffnOutput = { 0.9, 1.4 }; 

        double[] residual = new double[2];
        double[] norm = new double[2];

        for (int i = 0; i < 2; i++)
        {
            residual[i] = ffnOutput[i] + input[i];
            
            
        }

        double mean = (residual[0] + residual[1]) / 2.0;

        //Variance
        double varience = ((Math.Pow(residual[0] - mean, 2) + Math.Pow(residual[1] - mean, 2)) / 2.0);

        //Normalization
        double epsilon = 1e-5;

        for (int i = 0; i < 2; i++)
        {
            norm[i] = (residual[i] - mean) / Math.Sqrt(varience + epsilon);
            Console.WriteLine(norm[i]);
        }
    }
*/


    //RNN Testing
    /* 
   public static void Main()
    {
        // Inputs
        double input1 = 1;
        double input2 = 0;

        // Weights
        double inputToHiddenWeight = 0.5; // Input to Hidden
        double hiddenToHiddenWeight = 0.4; // Recurrent Hidden to Hidden
        double hiddenToOutputWeight = 0.8; // Hidden to Output

        // Initial hidden state
        double previousHiddenState = 0;

        double learningRate = 0.1;

        // Biases
        double hiddenBias = 0.1;
        double outputBias = 0.2;

        // Actual (Target) Output
        double actualOutput = 1;

        // Forward Pass
        double currentHiddenState = Math.Tanh(inputToHiddenWeight * input1 + hiddenToHiddenWeight * previousHiddenState + hiddenBias);
        double predictedOutput = hiddenToOutputWeight * currentHiddenState + outputBias;

        // Loss Calculation using Mean Squared Error (MSE)
        double loss = 0.5 * Math.Pow(predictedOutput - actualOutput, 2);

        Console.WriteLine($"Hidden State: {currentHiddenState}");
        Console.WriteLine($"Predicted Output: {predictedOutput}");
        Console.WriteLine($"Loss: {loss}");

        double gradientLoss = predictedOutput - actualOutput;
        double gradientOutw = (gradientLoss / predictedOutput) * currentHiddenState;
        double gradoutputBias = gradientLoss / predictedOutput;

        Console.WriteLine(gradientLoss);
        Console.WriteLine(gradoutputBias);
        Console.WriteLine(gradoutputBias);

        //gradient hidden state:
        double  gradhiddenstate = gradientLoss * hiddenToOutputWeight; 
        Console.WriteLine(gradhiddenstate);

        //deritave of tahn activation 
        double tahnDerivative = 1 - Math.Pow(currentHiddenState, 2);
        Console.WriteLine("Test:"+tahnDerivative);

        //grradient for hidden to hidden weight 

        double gradHiddentToHiddenweight = gradhiddenstate * tahnDerivative * previousHiddenState;
        Console.WriteLine(gradHiddentToHiddenweight);

        //gradient for hidden bias:

        double gradHiddenBias = gradhiddenstate * tahnDerivative;
        Console.WriteLine(gradHiddenBias);

        double newInputToHiddenWeight = inputToHiddenWeight - (learningRate * tahnDerivative);
        double newHiddenToHiddenWeight = hiddenToHiddenWeight;
        double newHiddenBias = hiddenBias - (learningRate * -0.2108);
        double newOutputBias = outputBias - (learningRate * -0.5882);

        Console.WriteLine(newInputToHiddenWeight);
        Console.WriteLine(newHiddenToHiddenWeight);
        Console.WriteLine(newHiddenBias);
        Console.WriteLine(newOutputBias);


        currentHiddenState = Math.Tanh(inputToHiddenWeight * input1 + hiddenToHiddenWeight * previousHiddenState + hiddenBias);
        predictedOutput = hiddenToOutputWeight * currentHiddenState + outputBias;
        loss = 0.5 * Math.Pow(learningRate - predictedOutput, 2);
        Console.WriteLine(predictedOutput);
        Console.WriteLine(currentHiddenState);
        Console.WriteLine(loss);
    }

    */

    //start of RNN model testing and encoded inputs  with some writing animation. Basic chatbot with some saving vocabs.
    /***************************************************************************
     ***************************************************************************

    //dictionary for encoded word inputs
   public static Dictionary<string, int> vocabulary = new Dictionary<string, int>
    {

        { "hello", 0 },
        { "hi", 1 },
        { "goodbye", 2 },
        { "how", 3 },
        { "are", 4 },
        { "you", 5 },
        { "name", 6 },
        { "what's", 7 },
        {"your",8 },
        {"whats",9 },
        
     
    };
    //testing
    static void Main()
    {
        
        Console.WriteLine("Hello, say last chat to see previous message, or clear memory to clear them.");

        //takes userinput, then runs the encode method to get int values, then writes back

        string filePath = "C:\\Users\\brand\\OneDrive\\Desktop\\Neuralnetworktesting\\Neuralnetworktesting\\vocab.json";

        if (File.Exists(filePath))
        {
            string existingFile = File.ReadAllText(filePath);
            vocabulary = JsonConvert.DeserializeObject<Dictionary<string, int>>(existingFile) ?? new Dictionary<string, int>();
        }

      
            string convopath = "C:\\Users\\brand\\OneDrive\\Desktop\\Neuralnetworktesting\\Neuralnetworktesting\\convo.txt";
            List<string> conversations = new List<string>();

            if (File.Exists(convopath))
            {
                conversations = File.ReadAllLines(convopath).ToList();
                Console.WriteLine("Previous conversations loaded.");
            }
            else
            {
                Console.WriteLine("No previous conversations found.");
            }
        

        String userinput = Console.ReadLine().ToLower();
        List<int> encoded = EncodeInput(userinput);
        Console.WriteLine("Encoded input: "+string.Join(",", encoded));
        string chatbot = "";



        if (encoded.Contains(0) || encoded.Contains(1))
        {
            chatbot = "Hello";
        }

        else if(encoded.Contains(3) && encoded.Contains(4) && encoded.Contains(5)) 
        {
            chatbot = "How are you?";
        }

        else if(encoded.Contains(7) || encoded.Contains(9) && encoded.Contains(8) && encoded.Contains(6))
        {
            chatbot = "I dont have a name, im a Ai language model.";
        }

        else if(encoded.Contains(-1))
        {
            Console.WriteLine("Sorry I dont know this word, can you say it again so I can log it?: ");
         
            string newword = Console.ReadLine().ToLower();
            
            if (vocabulary.ContainsKey(newword))
            {
                chatbot = chatbot + "Thanks I already know this word";
            }
            else
            {   vocabulary.Add(newword, vocabulary.Count);
                string json = JsonConvert.SerializeObject(vocabulary, Formatting.Indented);
                File.WriteAllText(filePath, json); // Save the updated vocabulary
                chatbot = chatbot + "Added word, thank you!";
                
            }

        }

        else
        {
            chatbot = "Sorry, im not sure how to respond to this yet.";
        }

        if (userinput.Contains("last chat"))
        {
            var lastUserInput = conversations
     .LastOrDefault(line => line.Trim().StartsWith("User:", StringComparison.OrdinalIgnoreCase));
            if (lastUserInput != null)
            {
                chatbot = ($"Your last question was: {lastUserInput.Substring(6)}"); // Remove "User: " prefix
            }
            else
            {
                chatbot = ("I don't remember your last question.");
            }
        }

        if(userinput.Contains("clear memory"))
        {
            conversations.Clear();
            File.WriteAllText(convopath, string.Empty);
            chatbot = "Memory wiped.";
        }


        //checks confidence based on words known vs provided. excludes -1 or unknown. 
        double confidence = (double)encoded.Count(e => e !=-1)/encoded.Count;
        
            //Types like chatgpt
            foreach (char c in chatbot)
            {
                Console.Write(c);
                
                Thread.Sleep(50);
            }
        
           

        if (vocabulary.Count > 0)
        {
            conversations.Add($"Time of chat: {DateTime.Now.ToString()}");
            conversations.Add($"User: {userinput}");
            conversations.Add($"Chatbot: {chatbot}");
            conversations.Add($"Confidence: {confidence:P}\n");
        }

        
        if (conversations != null)
        {
            File.WriteAllLines(convopath, conversations);
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
    */


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

