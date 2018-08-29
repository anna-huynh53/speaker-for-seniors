"""
Create virtual environment with:
    virtualenv env
    source env/bin/activate
Make sure Google credentials are set:
    export GOOGLE_APPLICATION_CREDENTIALS=
    "/home/anna/Documents/speaker-for-seniors/speaker-for-seniors-9f5d66f6c57b.json"
Example usage:
    python speaker.py --audio-file-path resources/filename.raw 
"""

#!/usr/bin/env python

import argparse
import io
import uuid
import time
import datetime
import json

# for transcribing using Google Cloud Speech API
from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types
# for finding intent and entities from Dialogflow API
import dialogflow_v2 as dialogflow
# to create json format from Dialogflow API call output
from google.protobuf.json_format import MessageToJson


# taken from https://github.com/GoogleCloudPlatform/python-docs-samples/speech/cloud-client
# converts speech-to-text for local file by calling Google Speech API
def transcribe_file(speech_file):  
    client = speech.SpeechClient()

    # [START speech_python_migration_sync_request]
    # [START speech_python_migration_config]
    with io.open(speech_file, 'rb') as audio_file:
        content = audio_file.read()

    audio = types.RecognitionAudio(content=content)
    config = types.RecognitionConfig(
        encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code='en-US')
    # [END speech_python_migration_config]

    # [START speech_python_migration_sync_response]
    response = client.recognize(config, audio)
    # [END speech_python_migration_sync_request]
    
    # each result is for a consecutive portion of the audio 
    # iterate through them to get the transcripts for the entire audio file
    for result in response.results:
        # the first alternative is the most likely one for this portion
        print(u'Transcript: {}'.format(result.alternatives[0].transcript))
    # [END speech_python_migration_sync_response]
    
    return result.alternatives[0].transcript


# converts speech-to-text for files stored on Google Cloud by calling Google
# Speech API
def transcribe_gcs(gcs_uri):
    client = speech.SpeechClient()

    # [START speech_python_migration_config_gcs]
    audio = types.RecognitionAudio(uri=gcs_uri)
    config = types.RecognitionConfig(
        encoding=enums.RecognitionConfig.AudioEncoding.FLAC,
        sample_rate_hertz=16000,
        language_code='en-US')
    # [END speech_python_migration_config_gcs]

    response = client.recognize(config, audio)
    
    # each result is for a consecutive portion of the audio 
    # iterate through them to get the transcripts for the entire audio file
    for result in response.results:
        # the first alternative is the most likely one for this portion
        print(u'Transcript: {}'.format(result.alternatives[0].transcript))
    
    return result.alternatives[0].transcript


# detects the intent and entities from given text
# using same session_id between requests allows continuation of the conversation
def detect_intent_texts(project_id, session_id, text, language_code):
    session_client = dialogflow.SessionsClient()

    session = session_client.session_path(project_id, session_id)
    print('Session path: {}\n'.format(session))

    text_input = dialogflow.types.TextInput(
        text=text, language_code=language_code)

    query_input = dialogflow.types.QueryInput(text=text_input)

    response = session_client.detect_intent(
        session=session, query_input=query_input)
        
    return response.query_result


def sort_query(result):
    # timestamp of request
    curr_time = time.time()
    timestamp = datetime.datetime.fromtimestamp(curr_time).strftime('%Y-%m-%d %H:%M:%S')
     
    query_text = result.query_text
    query_intent = result.intent.display_name
    query_confidence = result.intent_detection_confidence
    query_fulfillment = result.fulfillment_text
    
    # convert to json to access parameters
    jsonObj = MessageToJson(result)
    data = json.loads(jsonObj)
    query_verb = data['parameters']['verb'][0]
    query_object = data['parameters']['object'][0]
    
    print('=' * 20)
    print('Query text: {}'.format(query_text))
    print('Detected intent: {} (confidence: {})\n'.format(
        query_intent,
        query_confidence))
    print('Verb: {}'.format(query_verb))
    print('Object: {}'.format(query_object))
    print('Fulfillment text: {}\n'.format(query_fulfillment))
       
    # TODO: if the intent is not clear, ask user to rephrase to ensure accuracy?
    # TODO: what type of Google database system to use?
    #       store each query in separate file or in 1 file with table?
    
    if (query_intent == 'event.detect'):
        f = open("queries.txt", "a")
        f.write(query_text + " " + timestamp + "\n")
    #else if (intent == 'event.question')
       
    return


if __name__ == '__main__':
    project_id = 'speaker-for-seniors'

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '--session-id',
        help='Identifier of the DetectIntent session. '
        'Defaults to a random UUID.',
        default=str(uuid.uuid4()))
    parser.add_argument(
        '--language-code',
        help='Language code of the query. Defaults to "en-US".',
        default='en-US')
    parser.add_argument(
        '--audio-file-path',
        help='Path to the audio file.',
        required=True)

    args = parser.parse_args() 
        
    # speech-to-text call 
    # returns transcript and confidence level of given audio file
    if args.audio_file_path.startswith('gs://'):
        text = transcribe_gcs(args.audio_file_path)
    else:
        text = transcribe_file(args.audio_file_path)
       
    # TODO: speech-to-text from streaming input
    # TODO: if the confidence level is below a threshold, ask user to repeat   
       
    # dialogflow call
    # detects intent, entities and confidence of given text
    result = detect_intent_texts(project_id, args.session_id, text, 
        args.language_code)
        
    sort_query(result)

