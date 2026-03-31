from utils import load_deepstack_model

model = load_deepstack_model("cpu")

# 查看 wrapper 有哪些属性
print("=== DeepStackModelWrapper 的属性 ===")
for name in dir(model):
    if not name.startswith("_"):
        print(f"  {name}")

print("\n=== nn.Module 子模块 ===")
for name, module in model.named_children():
    print(f"  {name}: {type(module).__name__}")

print("\n=== 所有可训练参数名 ===")
for name, param in model.named_parameters():
    if "deepstack" in name.lower() or "connector" in name.lower():
        print(f"  {name}: shape={param.shape}, requires_grad={param.requires_grad}")