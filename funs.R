# function: load from JSON
load_from_json = function(file) {
	# load data
	json_data = rjson::fromJSON(file=file)
	# combinations
	sources = c("standard","adversarial")
	folds = c("train","valid","test")
	labels = c("bad","good")
	rounds = "1" # hard coded using only round 1
	combos = expand.grid(source=sources,fold=folds,label=labels,
	round=rounds)
	# loop over each combo to make dataframe
	list_of_dfs = lapply(1:nrow(combos),function(i) {
		# coordinates
		source = combos$source[i]
		fold = combos$fold[i]
		label = combos$label[i]
		round = combos$round[i]
		# construct dataframe
		df_data = json_data[[source]][[fold]][[round]][[label]]
		df_data = lapply(df_data,function(o) {
			data.frame(text=o$text,labels=o$labels,
			episode_done=o$episode_done)
		})
		df = do.call(rbind,df_data)
		# append combo information
		df$source = source
		df$fold = fold
		df$round = round
		df
	})
	return(do.call(rbind,list_of_dfs)) # put together
}

# function: take sample from dataframe
sample_from_df = function(x,n,bad_prevalence=NULL) {
	# always sample with replacement
	replace = TRUE
	# stratify by label if proportion specified
	if(is.null(bad_prevalence)) {
		sampled = sample(1:nrow(x),size=n,replace=replace)
	} else {
		# group IDs by label
		ids_bad = which(x$labels=="__notok__")
		ids_good = which(x$labels=="__ok__")
		# determine how many to sample per class
		size_bad = floor(n*bad_prevalence)
		size_good = n-size_bad
		# do stratified sampling
		sampled_bad = sample(ids_bad,size=size_bad,replace=replace)
		sampled_good = sample(ids_good,size=size_good,replace=replace)
		sampled = c(sampled_bad,sampled_good)
	}
	return(x[sampled,])
}

# function: convert to parlai format
convert_to_parlai_text = function(x,file) {
	# open sink
	sink(file)
	# write per row
	sapply(1:nrow(x),function(i) {
		# get fields
		text = x$text[i]
		labels = x$labels[i]
		episode_done = ifelse(x$episode_done[i],"True","False")
		# put together and write
		line_to_write = paste("text:",text,"\t","labels:",labels,
		"\t","episode_done:",episode_done,sep="")
		cat(line_to_write)
		cat("\n")
	})
	# close sink
	sink()
	return(invisible(NULL))
}

# function: corrupt labels
corrupt_labels = function(x,corrupt_rate,corrupt_action) {
	# fully corrupt vectors
	x_flip = 1-x
	x_lazy = rbinom(length(x),size=1,prob=0.5)
	# which to corrupt
	corrupted = rbinom(length(x),size=1,prob=corrupt_rate)
	# apply corruption
	if(corrupt_action=="flip") {
		x_corrupt = ifelse(corrupted==1,x_flip,x)
	} else if(corrupt_action=="lazy") {
		x_corrupt = ifelse(corrupted==1,x_lazy,x)
	} else {
		stop("corrupt type not recognized")
	}
	return(x_corrupt)
}

# function: get ratings
get_ratings = function(x,num_users,num_users_per_utter,troll_prevalence,
helper_corrupt_rate,troll_corrupt_rate,corrupt_action) {
	# get gold labels
	gold_labels = ifelse(x$labels=="__ok__",0,1)
	# create synthetic users
	num_trolls = floor(num_users*troll_prevalence)
	num_helpers = num_users-num_trolls
	user_type = c(rep("helper",times=num_helpers),
	rep("troll",times=num_trolls))
	# perform corruption of labels
	corrupt_rate = ifelse(user_type=="troll",troll_corrupt_rate,
	helper_corrupt_rate)
	matrix_of_labels = sapply(corrupt_rate,function(r) {
		corrupt_labels(gold_labels,corrupt_rate=r,
		corrupt_action=corrupt_action)
	}) # complete matrix of labels
	# generate missingness mask
	num_missing = num_users-num_users_per_utter
	missing_mask = t(replicate(nrow(x),{
		to_be_shuffled = c(rep(FALSE,num_users_per_utter),
		rep(TRUE,num_missing))
		sample(to_be_shuffled)
	}))
	# apply missingness to matrix of labels
	matrix_of_labels = replace(matrix_of_labels,missing_mask,NA)
	return(matrix_of_labels)
}

# function: determine clusters with latent variable modeling
cluster_lvm = function(x) {
	# remove useless users
	users_to_keep = apply(x,2,function(v) {
		how_many_unique = unique(v[!is.na(v)])
		length(how_many_unique)>1
	})
	# model fitting
	fit = mirt::mdirt(as.data.frame(x[,users_to_keep]),2)
	# get predictions
	predicted_prob = mirt::fscores(fit,method="EAP")[,1]
	predicted_bin = round(predicted_prob)
	return(setNames(predicted_prob,predicted_bin))
}

# function: cluster by majority vote
cluster_baseline = function(x) {
	# get predictions
	predicted_bin = apply(x,1,function(v) {
		nonmissing_vector = v[!is.na(v)]
		count_safe = sum(nonmissing_vector==0)
		count_unsafe = sum(nonmissing_vector==1)
		ifelse(count_safe>=count_unsafe,"__ok__","__notok__")
	})
	return(predicted_bin)
}

# function: imputation metrics
imp_metrics = function(gold,pred) {
	# make binary integer format
	if(!is.numeric(gold)) {
		gold_int = ifelse(gold=="__ok__",0,1)
	} else {
		gold_int = gold
	}
	if(!is.numeric(pred)) {
		pred_int = ifelse(pred=="__ok__",0,1)
	} else {
		pred_int = pred
	}
	# compute metrics
	confusion_table = table(gold=gold,pred=pred)
	accuracy = mean(gold_int==pred_int)
	recall = mean(pred_int[gold_int==1]==1) # sensitivity
	precision = mean(gold_int[pred_int==1]==1) # positive predictive value
	f1 = 2*recall*precision/(recall+precision) # F1 score
	metrics = list(confusion_table=confusion_table,accuracy=accuracy,f1=f1,
	precision=precision,recall=recall) # put together
	return(metrics)
}

# single replicate run
if(FALSE) {
	set.seed(2254)
	with(new.env(),{
		# scenario
		troll_prevalence = 0.7
		helper_corrupt_rate = 0.05
		troll_corrupt_rate = 0.8
		corrupt_action = "flip"
		# get safety data
		safety_data = load_from_json("./single_turn_safety.json")
		# sample training data
		dat = sample_from_df(safety_data,n=200,bad_prevalence=0.5)
		# create synthetic users and labels
		mol = get_ratings(dat,num_users=50,num_users_per_utter=5,
		troll_prevalence=troll_prevalence,
		helper_corrupt_rate=helper_corrupt_rate,
		troll_corrupt_rate=troll_corrupt_rate,
		corrupt_action=corrupt_action)
		# make predictions with baseline
		pred_baseline = cluster_baseline(mol)
		print(imp_metrics(gold=dat$labels,pred=pred_baseline))
		# make predictions with proposed
		pred_proposed = cluster_lvm(mol)
		pred_proposed_a = ifelse(round(pred_proposed)==0,"__ok__",
		"__notok__")
		pred_proposed_b = ifelse(round(pred_proposed)==1,"__ok__",
		"__notok__")
		print(imp_metrics(gold=dat$labels,pred=pred_proposed_a))
	})
}
