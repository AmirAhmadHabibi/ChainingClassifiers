import super_words_builder as sw_builder
import model_saver
import dimension_reducer

model_saver.save_chi_w2vs()

sw_builder.save_classifier_nouns()
sw_builder.save_time_stamps()
sw_builder.build_super_words()

model_saver.w2v_remove_non_superword()

dimension_reducer.reduce_LDA()
dimension_reducer.reduce_LDA(threshold=1950)

