
class ComposeAll:
    def __init__(self,transforms=[]) -> None:
        self.transforms = transforms
    def __call__(self, input):
        for tform in self.transforms:
            input = tform(input)    
        return input
    def __getitem__(self,idx):
        return self.transforms[idx]