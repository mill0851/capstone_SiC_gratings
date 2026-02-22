from phase1_preprocessing import dataset, train_loader, val_loader
from models_phase1.MLPModel import*
from util.eval_plot import*

mlp_model = MLPModel()
history = train_mlp(mlp_model, train_loader, val_loader, epochs=325)

# checkpoint = torch.load("mlp_final.pth", map_location="cpu")
# model.load_state_dict(checkpoint["model_state_dict"])
# history = checkpoint["history"]
# model.eval()  # ready for inference

plot_losses(history)
plot_lambda_predictions(mlp_model, val_loader, dataset)
plot_Q_predictions(mlp_model, val_loader)

torch.save({
    "model_state_dict": mlp_model.state_dict(),
    "history": history
}, "mlp_final.pth")


