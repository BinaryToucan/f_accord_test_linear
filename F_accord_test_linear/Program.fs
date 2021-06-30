// Learn more about F# at http://docs.microsoft.com/dotnet/fsharp

open System
open System
open Accord.MachineLearning
open Accord.MachineLearning.VectorMachines.Learning
open Accord.Statistics.Models.Regression.Linear
open FSharp.Data

open Accord.Statistics.Models.Regression
open Accord.Statistics.Models.Regression.Fitting
open Accord.DataSets
open Accord.MachineLearning.DecisionTrees
let traStep = 100
let leaRate = 0.0001f
let disStep = 10

// Sample data
let trX = [|12.3999; 14.3; 14.5; 14.8999; 16.1; 16.899; 16.5; 15.399; 17.0; 17.899;
              18.7999; 20.2999; 22.3999; 19.32; 15.5; 16.7|]
let trY = [|11.1999; 12.5; 12.699; 13.1; 14.100; 14.8; 14.39;
              13.3999; 14.8999; 15.6; 16.3999; 17.700; 19.60; 16.89; 14.0; 14.6|]

//let wei = [0.3; 0.25]
let ols = new OrdinaryLeastSquares()
let regres = ols.Learn(x = trX, y = trY)
let test = regres.Transform(12.32);

printf "X: 12.32, Y:%f" test