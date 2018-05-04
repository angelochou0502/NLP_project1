# NLP_project1

Fine-grained Sentiment Analysis on Financial Microblogs  

## Description

This project classifies the texts in twitter to one of three types(Bullish/Bearish/Natural)  

## Data

both training and testing json files are in the data/ directory  
example of data form:  
```
{
	"tweet":"downgrades $SON $ARI $GG $FLTX $WMC $MFA $IVR $CMI $PCAR $QLIK $AFOP $UNFI #stocks #investing #tradeideas",
	"target":"$PCAR",
	"snippet":"downgrade",
	"sentiment":-0.463
}
```
We set setiment < -0.2 to be the sign of 'Bearish', > 0.2 to be 'Bullsih' and 'Natural' otherwise.

## Running the tests

sh train&predict.sh (directory where to put trained rnn model)   
output: micro-f1 score and macro-f1 score
