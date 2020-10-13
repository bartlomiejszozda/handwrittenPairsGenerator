# thesis project:
Thesis summary (only in Polish)
https://github.com/bartlomiejszozda/handwrittenPairsGenerator/blob/master/thesisDocument/StreszczeniePracyInzynierskiej_BartlomiejSzozda.pdf  
Thesis (only in Polish)
https://github.com/bartlomiejszozda/handwrittenPairsGenerator/blob/master/thesisDocument/praca_inzynierska_BartlomiejSzozda.pdf  


Current GAN Network was changed and is quite different than that described in thesis document.

Continue GAN network learning:  
cd REPO/mainNetworks/gans  
jupyter-notebook  
run all cells

Start GAN network learning from the beginning:  
cd REPO/mainNetworks/gans  
jupyter-notebook  
Change variable 'continueWorking' to 'False'.   
Change variable 'dataDir' to what you want.   
Run all cells.  

You can ran GAN network learning from the console:  
cd REPO/mainNetworks/gans  
jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --inplace --execute gans3.ipynb

Check if data generation works (actually not need to run this because generation output is available for GAN Network):  
cd REPO  
python3 ./test/performance/shortGenerationPerformanceTest.py  
You can also check long tests:  
cd REPO/test   
python3 runAndLog_LongPerformanceTest.py  

You can enable pre-commit hooks to protect all new commits by running shortGenerationPerformanceTest:  
git config core.hooksPath hooks  


Thanks to the authors of the following projects:  
https://medium.com/ai-society/gans-from-scratch-1-a-deep-introduction-with-code-in-pytorch-and-tensorflow-cb03cdcdba0f  
https://github.com/Grzego/handwriting-generation  
https://github.com/githubharald/SimpleHTR  
https://github.com/Arlen0615/Convert-own-data-to-MNIST-format  
https://github.com/myleott/mnist_png/blob/master/convert_mnist_to_png.py  
