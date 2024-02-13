{\rtf1\ansi\ansicpg1252\cocoartf2759
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fnil\fcharset0 HelveticaNeue-Bold;\f1\fnil\fcharset0 HelveticaNeue-Medium;\f2\fnil\fcharset0 Menlo-Regular;
}
{\colortbl;\red255\green255\blue255;\red25\green28\blue31;\red255\green255\blue255;\red29\green29\blue29;
\red89\green89\blue90;\red240\green241\blue245;}
{\*\expandedcolortbl;;\cssrgb\c12941\c14510\c16078;\cssrgb\c100000\c100000\c100000;\cssrgb\c14902\c14902\c14902;
\cssrgb\c42353\c42353\c42745;\cssrgb\c95294\c95686\c96863;}
{\*\listtable{\list\listtemplateid1\listhybrid{\listlevel\levelnfc23\levelnfcn23\leveljc0\leveljcn0\levelfollow0\levelstartat1\levelspace360\levelindent0{\*\levelmarker \{disc\}}{\leveltext\leveltemplateid1\'01\uc0\u8226 ;}{\levelnumbers;}\fi-360\li720\lin720 }{\listname ;}\listid1}}
{\*\listoverridetable{\listoverride\listid1\listoverridecount0\ls1}}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\deftab720
\pard\pardeftab720\sa330\partightenfactor0

\f0\b\fs48 \AppleTypeServices\AppleTypeServicesF65539 \cf2 \cb3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 Neural Networks\
\pard\pardeftab720\sa270\partightenfactor0

\f1\b0\fs32 \AppleTypeServices\AppleTypeServicesF65539 \cf4 \strokec4 Neural networks can be constructed using the\'a0
\f2\fs25\fsmilli12560 \AppleTypeServices \cf5 \cb6 \strokec5 torch.nn
\f1\fs32 \AppleTypeServices\AppleTypeServicesF65539 \cf4 \cb3 \strokec4 \'a0package.\
Now that you had a glimpse of\'a0
\f2\fs25\fsmilli12560 \AppleTypeServices \cf5 \cb6 \strokec5 autograd
\f1\fs32 \AppleTypeServices\AppleTypeServicesF65539 \cf4 \cb3 \strokec4 ,\'a0
\f2\fs25\fsmilli12560 \AppleTypeServices \cf5 \cb6 \strokec5 nn
\f1\fs32 \AppleTypeServices\AppleTypeServicesF65539 \cf4 \cb3 \strokec4 \'a0depends on\'a0
\f2\fs25\fsmilli12560 \AppleTypeServices \cf5 \cb6 \strokec5 autograd
\f1\fs32 \AppleTypeServices\AppleTypeServicesF65539 \cf4 \cb3 \strokec4 \'a0to define models and differentiate them. An\'a0
\f2\fs25\fsmilli12560 \AppleTypeServices \cf5 \cb6 \strokec5 nn.Module
\f1\fs32 \AppleTypeServices\AppleTypeServicesF65539 \cf4 \cb3 \strokec4 \'a0contains layers, and a method\'a0
\f2\fs25\fsmilli12560 \AppleTypeServices \cf5 \cb6 \strokec5 forward(input)
\f1\fs32 \AppleTypeServices\AppleTypeServicesF65539 \cf4 \cb3 \strokec4 \'a0that returns the\'a0
\f2\fs25\fsmilli12560 \AppleTypeServices \cf5 \cb6 \strokec5 output
\f1\fs32 \AppleTypeServices\AppleTypeServicesF65539 \cf4 \cb3 \strokec4 .\
It is a simple feed-forward network. It takes the input, feeds it through several layers one after the other, and then finally gives the output.\
A typical training procedure for a neural network is as follows:\
\pard\tx220\tx720\pardeftab720\li720\fi-720\partightenfactor0
\ls1\ilvl0
\fs24 \AppleTypeServices\AppleTypeServicesF65539 \cf4 \cb3 \kerning1\expnd0\expndtw0 \outl0\strokewidth0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec4 Define the neural network that has some learnable parameters (or weights)\cb1 \
\ls1\ilvl0\cb3 \kerning1\expnd0\expndtw0 \outl0\strokewidth0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec4 Iterate over a dataset of inputs\cb1 \
\ls1\ilvl0\cb3 \kerning1\expnd0\expndtw0 \outl0\strokewidth0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec4 Process input through the network\cb1 \
\ls1\ilvl0\cb3 \kerning1\expnd0\expndtw0 \outl0\strokewidth0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec4 Compute the loss (how far is the output from being correct)\cb1 \
\ls1\ilvl0\cb3 \kerning1\expnd0\expndtw0 \outl0\strokewidth0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec4 Propagate gradients back into the network\'92s parameters\cb1 \
\ls1\ilvl0\cb3 \kerning1\expnd0\expndtw0 \outl0\strokewidth0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec4 Update the weights of the network, typically using a simple update rule:\'a0
\f2\fs25\fsmilli12560 \AppleTypeServices \cf5 \cb6 \strokec5 weight\cf4 \cb3 \strokec4 \'a0\cf5 \cb6 \strokec5 =\cf4 \cb3 \strokec4 \'a0\cf5 \cb6 \strokec5 weight\cf4 \cb3 \strokec4 \'a0\cf5 \cb6 \strokec5 -\cf4 \cb3 \strokec4 \'a0\cf5 \cb6 \strokec5 learning_rate\cf4 \cb3 \strokec4 \'a0\cf5 \cb6 \strokec5 *\cf4 \cb3 \strokec4 \'a0\cf5 \cb6 \strokec5 gradient
\f1\fs24 \AppleTypeServices\AppleTypeServicesF65539 \cf4 \cb1 \strokec4 \
}