import json

examples = [
    {
        "user_message": "What is Troy's LinkedIn?",
        "response": "Troy's LinkedIn can be found towards the top of his resume:\\nlinkedin.com/in/troy-fulton",  # noqa: E501
        "sources": [
            "resume.pdf",
        ],
    },
    {
        "user_message": "Does Troy have any experience with big data?",
        "response": 'According to Troy\'s resume, he developed "Big Data infrastructure using Kubernetes" while working at Aspen Insights from June 2020 - Present.',  # noqa: E501
        "sources": [
            "resume.pdf",
        ],
    },
]

examples_json = [
    {
        "user_message": example["user_message"],
        "response": json.dumps(example["response"]),
        "sources": json.dumps(example["sources"]),
    }
    for example in examples
]
