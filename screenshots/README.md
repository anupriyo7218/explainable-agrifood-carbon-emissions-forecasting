The Streamlit dashboard has been fully implemented and verified in a local environment.

The bar plot in SHAP_Explainabiity ranks each feature by its mean absolute SHAP value across every test-set prediction — a single number per feature representing average impact magnitude, regardless of direction. Longer bars indicate features that, on average, move the model's prediction further from the baseline.

In Summary Plot (Beeswarm), each dot is one test-set prediction. Horizontal position shows that feature's SHAP value (impact on the predicted emission) for that specific row; color shows whether the feature's actual value was high (red) or low (blue). This reveals direction, which the bar plot above does not — e.g. whether high urban population consistently pushes predicted emissions up, down, or depends on other context.
