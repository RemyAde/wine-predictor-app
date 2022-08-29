from django.shortcuts import render
import pandas as pd
import numpy as np

def home(request):
    return render(request, 'predictor/home.html')

def results(request):

    FA = float(request.GET.get("FA"))
    VA = float(request.GET.get("VA"))
    CA = float(request.GET.get("CA"))
    RS = float(request.GET.get("RS"))
    CHO = float(request.GET.get("Ch"))
    FSO = float(request.GET.get("FSO"))
    TSO = float(request.GET.get("TSO"))
    DSO = float(request.GET.get("density"))
    PH = float(request.GET.get("pH"))
    SOO = float(request.GET.get("Su"))
    ALO = float(request.GET.get("Al"))


    test_pred_vals = np.array([FA,VA,CA,RS,CHO,FSO,TSO,DSO,PH,SOO,ALO]).reshape(1,-1)
    model = pd.read_pickle(r'c:\Users\USER\Desktop\Work\djangoStuffs\ml_django\ml\new.pickle')
    pred_results = model.predict(test_pred_vals)
    pred_results = pred_results[0]

    return render(request, 'predictor/results.html', {'pred_results':pred_results})

def customers(request):
    return render(request, "predictor/customers.html")


def amount(request):

    AVT = float(request.GET.get("AVT"))
    TA = float(request.GET.get("TA"))
    TW = float(request.GET.get("TW"))
    LM = float(request.GET.get("LM"))


    test_vals = np.array([AVT,TA,TW,LM]).reshape(1,-1)
    model = pd.read_pickle(r'c:\Users\USER\Desktop\Work\djangoStuffs\lmodel.pickle')
    pred_vals = model.predict(test_vals)
    

    return render(request, "predictor/yearly_amount.html", {'pred_vals':pred_vals})
