readme.txt for kNN.py

From a terminal window, in order for kNN.py file to run, must precede filename with 'python3' to run this python file.
The next two required arguments after the filename should be the Training dataset, followed by the Test dataset, in that order.
As the final argument, the desired k-value (k-nearest-neighbours) as an integer should follow lastly.

Follow format below:
python3 kNN.py wine-training wine-test 1

Errors will occur if dataset filenames don't match and cannot be found. An error will be presented if missing arguments 
  or if last argument for k-value is not of type integer.
  
In terminal window the kNN method will go to work and firstly display the K-value that has been chosen by the user running the program.
Following that will be the prediction of each test instance and the predicted classification made by the kNN Method.
Beside each prediction will be whether the prediction was correct or incorrect (will display what the test instance's class really was)
Lastly, a fraction of correct predictions vs total test instances will be displayed and the accuracy rate displayed also.


~Santino Gaeta
300305101 (Gaetasant)