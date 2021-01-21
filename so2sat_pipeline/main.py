import train_so2sat
import test_so2sat

def main():
    model = train_so2sat.train()
    test_so2sat.test(model)

if __name__ == "__main__":
    main()
