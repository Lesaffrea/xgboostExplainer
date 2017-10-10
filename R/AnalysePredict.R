# Common declaration
#
#   Input:  Table for one tree
#
#   Output: The table of leaves with the gain for each varaiable in the path. Made of:
#           Leaf = Node reference for the tree processed (not the ID)
#
#

BuildLeavesGain <-function(Tree, AllVariables ){
                  if(!is.data.frame(Tree))            {stop("Expect one data frame for tree")}
                  if(!is.character(AllVariables))     {stop("The variables name must be of type character")}
                  # buildleavepath
                  #   This function return the indexes of the splits from leaf to root in tree
                  #   LeaveRef refernce one ID and it is referenced only once in the full tree in
                  #   in Yes or No
                  #
                  #   Input:   The end leaf ID reference
                  #
                  #   Output : Vector of Node starting with leaf reference and going up to root. The leaf gain is
                  #            the end gain that is made of the gain across the path- We are in binary we work
                  #            logit / log odds
                  #
                  # Warning: Tree is in the callee environment!!!!
                  BuildLeafPath<-function(LeaveRef){
                    stopifnot(is.character(LeaveRef))            # We expect ID
                    pathback <-character(0)
                    curtree  <-strsplit(LeaveRef, "-")[[1]][1]
                    if( Tree$Node[Tree$ID == LeaveRef] == 0 )         { stop("The function can not start at the root of the tree")}
                    # Build temporary vector of tree without leaves
                    YesNoVec <-Tree %>%
                                  dplyr::select(Yes,No, ID) %>%
                                  dplyr::filter(!is.na(Yes))
                    currentnode <-LeaveRef
                    while(as.integer(strsplit(currentnode, "-")[[1]][2]) != 0){
                      pathback <-c(pathback, currentnode)
                      currentnode <-c(YesNoVec$ID[YesNoVec$Yes == currentnode],
                                      YesNoVec$ID[YesNoVec$No == currentnode])
                      stopifnot(length(currentnode) == 1)
                    }
                    return(c(pathback,paste0(curtree, "-0" )))
                  }
                  leavesnode <- sort(Tree$ID[Tree$Leave == TRUE])  #Not necessary to sort but easier to follow
                  #Build the gain variable reference table to give back
                  LeaveGain        <- as.data.frame(matrix(replicate((length(AllVariables) + 2)* length(leavesnode), 0),
                                                       nrow=length(leavesnode)))
                  names(LeaveGain) <- c("Leaf", "Intercept", AllVariables)
                  #Build a tree without leaves we pass to BuildLeafPath(), it will be on shallow copy as we do not do any
                  #change in BuildLeafPath()
                  for( indexleaf in  1:length(leavesnode)){
                       curleaf                    <- leavesnode[indexleaf]
                       LeaveGain$Leaf[indexleaf]  <-curleaf
                       allbranches    <-BuildLeafPath(curleaf)          #Get from leaf end to root, starting with leaf
                       for( inleafpath in 1:length(allbranches)){
                             # Process leaf that is the final gain, which is the log odd for this leaf = Intercept
                             # if not one then the names of the Tree$Feature must be in LeaveGain
                             if(inleafpath ==1){
                                            LeaveGain$Intercept[indexleaf] <- Tree$NodeWeight[Tree$ID==allbranches[inleafpath]]
                              }else{
                                            # Put the node gain associated with the varaiable name in the return table
                                            LeaveGain[indexleaf ,
                                                      which(names(LeaveGain) == Tree$Feature[Tree$ID == allbranches[inleafpath]])] <-
                                                          Tree$NodeWeight[Tree$ID==allbranches[inleafpath]]
                              }
                      }
                  }
                  return(LeaveGain)
}
#'
#'
#'
#' This function builds the breakdown of contribution to prediction in case binary prediction
#' for xgboost only.
#'
#' Thanks a lot to David Foster to show how to extract details information out of the structure
#' return by xgb.model.dt.tree() and the usage of predict() with option PredLeaf = TRUE
#'
#' Structure of one node in one tree:
#'       Attributes:
#'                   1. Weight = Original Quality (All)
#'                   2. Gain   = Weight * Correct Cover (All)
#'                   3. CorrectCover = Cover (Leave == TRUE), Sum of children Correct Cover (Split=Leave = FALSE)
#'
#' @export ExplainPredictionsXgboost
ExplainPredictionsXgboost<-function(xgboostmodel, datatopredict){
  # Data Declaration
  ModelSummary <-list( NumbTrees = integer(1),
                       NumbNodes = integer(1)
                     )
  # Check
  if( class(xgboostmodel) =="train") {xgboostmodel <- xgboostmodel$finalmodel}  #We support Caret
  if(!inherits(xgboostmodel,"xgb.Booster"))  {stop("Could not process the type of model - Expects xgb.Booster")}
  if( class(datatopredict) != "xgb.DMatrix") {stop("Expect data type xgb.Matrix , please use xgb.train for interface")}
  if(novariablename <-is.null(xgboostmodel$feature_names))   {warning("No variables name included in model will allocate from data")}
  if( novariablename ==TRUE){
          Variables <-attr(datatopredict,".Dimnames")[[2]]
  }else{
          Variables <-xgboostmodel$feature_names
  }
  # Start processing
  # 1. Extract the tree structure, we shall get all trees - Warning: give back one data table .....
  ModelInternal <-as.data.frame(xgb.model.dt.tree(
                                                  feature_names = Variables,
                                                  model        = xgboostmodel,
                                                  n_first_tree = xgboostmodel$best_ntreelimit - 1
                                                  ))
  # We build a sequence to go through the trees in one easier way
  ModelInternal$Sequence <- seq(from=1, to=nrow(ModelInternal), by=1)
  ModelSummary$NumbTrees <- max(ModelInternal$Tree) +1  # The tree starts at zero
  # Trees statistics
  # We can start all we have to extract the trees cover and gain preparation (important)
  # 1. We correct the cover as it have one issue
  ModelInternal <- ModelInternal %>%
                          dplyr::mutate( Leave        = ifelse( Feature == "Leaf", TRUE,FALSE),
                                         CorrectCover = Cover)
  # Get internal nodes and add the cover from the left and rigth, we go reverse as the cover comulate
  Splits <- ModelInternal$Sequence[ModelInternal$Leave == FALSE] # Set processing order
  Splits <- sort(Splits, decreasing=TRUE)
  ModelSummary$NumbNodes <-length(Splits)
  for(cursplits in Splits){
      ModelInternal$CorrectCover[cursplits] <-with(ModelInternal,{
                                                    CorrectCover[ID==Yes[cursplits]] +
                                                    CorrectCover[ID==No[cursplits]]})
  }
  # 2. We process the leaves we have to weight to add here
  ModelInternal$Weight <-with( ModelInternal, ifelse( Leave == TRUE,
                                                      Quality,
                                                      0))
  ModelInternal$Gain <-with( ModelInternal, ifelse( Leave == TRUE,
                                                      -Weight * CorrectCover,
                                                      0))
  #Transform in seprate trees  and then propagate the Gain , we have to recalcualte the sequence
  ModelInternal  <-split(ModelInternal, ModelInternal$Tree)
  ModelInternal  <-lapply(ModelInternal, function(oneTree){
                                   Splits <-seq(from=1, to=nrow(oneTree), by=1)
                                   Splits <-Splits[oneTree$Leave == FALSE]
                                   Splits <-sort(Splits, decreasing = TRUE)   # A way of going from bottom to top
                                   for(cursplits in Splits){
                                       leftTree  <-oneTree[which(with(oneTree,{ ID== Yes[cursplits] })),]
                                       rightTree <-oneTree[which(with(oneTree,{ ID== No[cursplits] })),]
                                       oneTree$Gain[cursplits] <- leftTree$Gain + rightTree$Gain
                                       oneTree$Weight[cursplits] <-with(oneTree,{-Gain[cursplits] / CorrectCover[cursplits]})
                                       oneTree$FatherWeight[which(with(oneTree,{ID == Yes[cursplits]}))]  <-
                                       oneTree$FatherWeight[which(with(oneTree, {ID==No[cursplits]}))]    <-oneTree$Weight[cursplits]
                                   }
                                   oneTree$FatherWeight[1] <- 0 #This is the root Setting
                                   oneTree$NodeWeight <- with(oneTree, { Weight - FatherWeight})  #The node contribution no the pathh weight
                            return(oneTree)
                })
    #At this stage we have each Tree in a list of trees we have to map the gain to the variables in Features
    #We have to scan the tree and then rebuild a new table the main variable used here is NodeWeight as it
    #the node contribution for one calculation (See details in the paper about Xgboost for the details of Hessian)
               AllGainTable <-list()
               for( curtree in 1:length(ModelInternal)){
                   AllGainTable[[curtree]] <-BuildLeavesGain( ModelInternal[[curtree]] ,Variables)
                   }
#    ModelInternal <-do.call(rbind,ModelInternal)
    back <-list(GainTable     =AllGainTable,
                InternalModel =ModelInternal,
                Summary       =ModelSummary)
  return(back)
}

#'
#'  BuildPredictionBreakdown
#'
#'   In this function we build the table of gain per observation to predict
#'
#'    @param treegain the GainTable given back by ExplainPredictionsXgboost
#'    @param xgbmodel
#'    @param datatopredict
#'    @param dataindex
#'
#'    Ouput:  The table of gain for the binary prediction
#'                         Index: The index given as input for the observation in the datapredict data table
#'                         Allvariable + Intercept ; The gain comulated over the trees as done in ExplainPredictionsXgboost
#'
#'    Note: We do a prediction to get the end leaf only, no new prediction
#'
#' @export BuildPredictionBreakdown
BuildPredictionBreakdown <-function( treegain,
                                        xgbmodel,
                                        datatopredict,
                                        dataindex){

        if( class(datatopredict) != "xgb.DMatrix") {stop("Expect data type xgb.Matrix , please use xgb.train for interface")}
        if( max(dataindex) > nrow(datatopredict))  {stop("Index of observation not in range of the data given")}
        #    We have to change the leaf of the treegain to fit with prediction
        tmp <-lapply(treegain,
                             function(onetree){ as.integer(unlist(lapply((strsplit(onetree$Leaf, "-")), function(two) {two[2]})))})
        for(index in 1:length(treegain)){
                                    treegain[[index]]$LeafInt <-tmp[[index]]
        }
        #    We use prediction with leaf rebuild one temporary data set
        allpredict <-predict(xgbmodel,
                             newdata=xgboost::slice(datatopredict,
                                            idxset= as.integer(dataindex)),
                             predleaf =TRUE)
        stopifnot(ncol(allpredict) == length(treegain))  # Security
        # Build the prediction gain table  add 1 for intercept
        backdf <- as.data.frame( matrix(replicate(((ncol(datatopredict) +1) * length(dataindex)), 0),   #Less leaf and Leafint
                                        nrow=length(dataindex)))
        # We put all names as one tree we remove leaf and Lieaf int
        ToRemove      <- -c(1,ncol(treegain[[1]]))
        names(backdf) <-names(treegain[[1]])[ToRemove]
        # We can now go througth the prediction and the treegain table we go prediction by prediction
        numbtree  <- length(treegain)
        for(curtopredict in 1:length(dataindex)){
                              curpredittree  <-allpredict[curtopredict,]
                              for( curtree in 1:numbtree){
                                    backdf[curtopredict,] <- backdf[curtopredict,] +      #We build the name based on tree gain
                                                             treegain[[curtree]][ treegain[[curtree]]$LeafInt ==allpredict[curtopredict,curtree],
                                                                                 ToRemove]

                              }
        }
        # At this stage we have all the gains by prediction
      return(cbind(Index=dataindex, backdf))
}

#
#
#
# back <-ExplainPredictionsXgboost(xgb.model,xgb.train.data)
# allgain <-BuildPredictionBreakdown( back$GainTable,
#                                     xgb.model,
#                                     xgb.train.data,
#                                     c(1,2,3,4,5))


