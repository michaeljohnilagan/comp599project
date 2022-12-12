# load stuff
source("./funs.R") # functions
safety_data = load_from_json(file="./single_turn_safety.json") # full dataset

# simulation factors
sim_scen = list(troll_prevalence=c(0.7,0.3),
troll_corrupt_rate=c(0.8,0.95),
corrupt_action=c("flip","lazy"),
proposed=c(TRUE,FALSE))

# constants
const = list(helper_corrupt_rate=0.05,n=200,num_users=50,
num_users_per_utter=5)

# set seeds
seeds = seq(937,by=1,length.out=5)

# function: run single replicate
run_sim_repl = function(troll_prevalence,troll_corrupt_rate,corrupt_action,
proposed) {
	# training, valid, test
	samp_train = sample_from_df(subset(safety_data,fold=="train"),200,
	bad_prevalence=0.5)
	samp_valid = sample_from_df(subset(safety_data,fold=="train"),24,
	bad_prevalence=0.5) # take another 24 from training set for validation
	samp_eval = subset(safety_data,fold=="valid"&source=="standard",
	bad_prevalence=NULL)
	# create matrix of labels
	matrix_of_labels = get_ratings(samp_train,num_users=const$num_users,
	num_users_per_utter=const$num_users_per_utter,
	troll_prevalence=troll_prevalence,
	helper_corrupt_rate=const$helper_corrupt_rate,
	troll_corrupt_rate=troll_corrupt_rate,corrupt_action=corrupt_action)
	# make predictions
	if(proposed) {
		pred = cluster_lvm(matrix_of_labels)
		pred_a = ifelse(round(pred)==0,"__ok__",
		"__notok__") # completion A
		metrics_a = imp_metrics(gold=samp_train$labels,pred=pred_a)
		pred_b = ifelse(round(pred)==1,"__ok__",
		"__notok__") # completion B
		metrics_b = imp_metrics(gold=samp_train$labels,pred=pred_b)
		if(metrics_a$accuracy>metrics_b$accuracy) {
			metrics = metrics_a
			which_completion = "a"
		} else {
			metrics = metrics_b
			which_completion = "b"
		} # choose between A and B, whichever is more accurate
		
	} else {
		pred = cluster_baseline(matrix_of_labels)
		metrics = imp_metrics(gold=samp_train$labels,pred=pred)
		which_completion = NA
	}
	# put together
	allthethings = list(samp_train=samp_train,samp_valid=samp_valid,
	samp_eval=samp_eval,matrix_of_labels=matrix_of_labels,metrics=metrics,
	which_completion=which_completion)
	if(proposed) {
		allthethings = c(allthethings,list(pred_a=pred_a),
		list(pred_b=pred_b))
	} else {
		allthethings = c(allthethings,list(pred=pred))
	}
	return(allthethings)
}

# work
resu = array(list(),sapply(sim_scen,length)) # initialize array
resu_tab = NULL # initialize dataframe
for(i1 in 1:length(sim_scen$troll_prevalence))
for(i2 in 1:length(sim_scen$troll_corrupt_rate))
for(i3 in 1:length(sim_scen$corrupt_action))
for(i4 in 1:length(sim_scen$proposed)) {
	# report
	message(Sys.time())
	troll_prevalence = sim_scen$troll_prevalence[[i1]]
	print(troll_prevalence)
	troll_corrupt_rate = sim_scen$troll_corrupt_rate[[i2]]
	print(troll_corrupt_rate)
	corrupt_action = sim_scen$corrupt_action[[i3]]
	print(corrupt_action)
	proposed = sim_scen$proposed[[i4]]
	print(proposed)
	# create scenario folder
	impute_method = ifelse(sim_scen$proposed[[i4]],"proposed",
	"baseline")
	if(!dir.exists(impute_method)) dir.create(impute_method)
	path = paste(impute_method,"/prev",i1,"corr",i2,"type",
	i3,sep="")
	dir.create(path)
	# work
	resu[[i1,i2,i3,i4]] = lapply(1:length(seeds),function(i) {
		# set seed
		set.seed(seeds[i])
		# create replicate folder
		path = paste(path,"/run",i,sep="")
		dir.create(path)
		message(path)
		# run replicate
		curr = run_sim_repl(troll_prevalence=troll_prevalence,
		troll_corrupt_rate=troll_corrupt_rate,
		corrupt_action=corrupt_action,proposed=proposed)
		# create files
		if(proposed) {
			df_a = curr$samp_train
			df_a$labels = curr$pred_a
			convert_to_parlai_text(df_a,file=paste(path,
			"/data_train-a.txt",sep=""))
			df_b = curr$samp_train
			df_b$labels = curr$pred_b
			convert_to_parlai_text(df_b,file=paste(path,
			"/data_train-b.txt",sep=""))
		} else {
			df = curr$samp_train
			df$labels = curr$pred
			convert_to_parlai_text(df,file=paste(path,
			"/data_train.txt",sep=""))
		} # train
		convert_to_parlai_text(curr$samp_valid,file=paste(path,
		"/data_valid.txt",sep="")) # valid
		convert_to_parlai_text(curr$samp_eval,file=paste(path,
		"/data_test.txt",sep="")) # test
		# return object
		c(curr$metrics,which_completion=curr$which_completion)
	})
	# tabular form
	resu_tab = rbind(resu_tab,
	with(sim_scen,{
		data.frame(troll_prevalence=troll_prevalence[i1],
		troll_corrupt_rate=troll_corrupt_rate[i2],
		corrupt_action=corrupt_action[i3],
		proposed=proposed[i4],
		run=1:5,
		annotation_train_acc=sapply(resu[[i1,i2,i3,i4]],
		function(o) {o$accuracy}))
	}))
}; Sys.time() # fill array

# create CSV file for results
write.csv(resu_tab,"resu_tab.csv",row.names=FALSE)

# save and end session
save.image("comp599proj.RData")
devtools::session_info()
