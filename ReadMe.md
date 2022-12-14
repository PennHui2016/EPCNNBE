## EPCNNBE
A base editor design tool for measuring editing efficiencies, outcome proportions and recommending the best protospacer for a given expected outcome.
We released a webserver tool for users to try our tool, see http://www.epcnnbe.com/. This repository stores the source codes for reproduce our results reported in our manucsript and the source codes for running our tool with command line.

# Part 1 required python packages for running sources codes
The codes programmed with python 3.8.8.Required packages include:
1. pandas
2. numpy
3. random
4. os
5. copy
6. csv
7. sys
8. torch 1.5.1
9. sklearn
10. Biopython
11. scipy

# Part 2 description of source code files
Before running the source codes, please download our network weights file 'be_weights.zip' via the link: https://drive.google.com/file/d/1yqnhpsEKsnY6L7Cv0yjcwOcNHVjLPYGA/view?usp=sharing.
Then, extracting the files from the compressed file to the fold ./be_weights/.

The file EPCNNBE_independent_tests.py can be run by command line of:

python EPCNNBE_independent_tests.py

Then, the results for independent tests can be found in the folder ./res/.

The file EPCNNBE_cross_platform.py is for reproducing the results of cross-platform tests of different tools, simply using command:

python EPCNNBE_cross_platform.py

The result files are also stored in the folder ./res/.

The file EPCNNBE_web.py is the main interface for trying our tool EPCNNBE. One can run the tool with the following command:

python EPCNNBE_web.py be_type fasta_file_path expect_outcome_sequence

where, three parameters are request including be_type fasta_file_path expect_outcome_sequence:

be_type --- string, specify the base editor type, 'abe' and 'cbe' are supported
fasta_file_path --- string, specify the fasta file containing the DNA sequence to be edited by a base editor. An example file see 'test_seq_SOD1.fa'. The DNA sequence should be no less than 30nt.
expect_outcome_sequence --- string, specify the expected outcome sequence. This parameter is optional, only filled if users expect to choose the best protospacer for editing the DNA and get the given expected outcome sequence. 

An example command:

python EPCNNBE_web.py 'abe' test_seq_SOD1.fa 			

Any problems or requesting source codes for reproducing results in our paper please contact 
    Hui Peng: hui.peng@ntu.edu.sg or cdph2009@163.com

                        
