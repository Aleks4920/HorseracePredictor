from argparse import ArgumentParser, Namespace
from typing import List
from predict_horses import predict_my_horse

# a CLI Application that when called asks for the names of the horses and predicts the winner

def main(args: Namespace) -> None:
    horses: List[str] = []
    
    while True:
        horse = input("Enter the name of the horse: ")
        if horse == "" or horse == "done" or horse == "exit":
            break
        horses.append(horse)
    
    
    predict_my_horse(horses)
    
if __name__ == '__main__':
    parser = ArgumentParser(description="Predict the winner")
    args = parser.parse_args()
    main(args)
    