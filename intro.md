## some stuff

### Neural Networks in Computer Vision

| What people think they will do                                                            | What they actually do                                                                            |
| ----------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| Build crazy neural nets                                                                   | Detect and correct imaging artifacts. Seriously... this is like half the job.                    |
| Build the next great machine learning system                                              | Getting the damn sensor to run reliably and cleanly.                                             |
| Deep learning, deep learning, deep learning, deep learning,                               | Spends hours on Figure 8 building data sets. Then hours more scrubbing the data.                 |
| deep learning, deep learning, deep learning, deep learning, deep learning, deep learning, | Moving gigs of data between S3 buckets.                                                          |
| deep learning, deep learning, deep learning, deep learning,                               | Build infrastracture to build, deploy, version, track, and validate models.                      |
| deep learning, deep learning, deep learning, deep learning, deep learning, deep learning, | Sit in boring meetings with product managers explaining precision and recall for the third time. |
| deep learning, deep learning,                                                             | `Getting matplotlib to work right.`                                                              |

### Data Problem
<b>Never Fake Data</b>

In practice, you will always find your self between unsuficient data for your project and a manager telling you "fake it" or "generate it". There are real cases when someone got an awesome idea to generate testing data. It will never work. There is always something (at least in big data), in real world, that one can never think about. So using real data (a lot of them) will always give you proper results, while doing anything on faked (or generated) data, will only lead you to something that is useless.