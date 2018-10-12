"""
Create virtual environment with:
    virtualenv env
    source env/bin/activate
Make sure Google credentials are set:
    export GOOGLE_APPLICATION_CREDENTIALS=
    "/home/anna/Documents/speaker-for-seniors/speaker-for-seniors-9f5d66f6c57b.json"
Install requirements:
    pip install -r requirements.txt
Example usage:
    python speaker.py --audio-file-path resources/filename.raw 
To record audio file:
    arecord "filename".raw --rate=16000 --format=S16
    
To connect the database:
Install the proxy client on your local machine:
    wget https://dl.google.com/cloudsql/cloud_sql_proxy.linux.amd64 -O cloud_sql_proxy
    chmod +x cloud_sql_proxy
Start the proxy:
    sudo mkdir /cloudsql; sudo chmod 777 /cloudsql
    ./cloud_sql_proxy -dir=/cloudsql &
"""

#!/usr/bin/env python

import argparse
import io
import os
import subprocess
import uuid
import time
import datetime
from datetime import datetime
from datetime import timedelta
import json
import pymysql
from sqlalchemy import create_engine

# for transcribing using Google Cloud Speech API
from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types
# for finding intent and entities from Dialogflow API
import dialogflow_v2 as dialogflow
# to create json format from Dialogflow API call output
from google.protobuf.json_format import MessageToJson
# to output speech using Google Cloud text-to-speech API
from google.cloud import texttospeech
                             
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
        print(u'Confidence: {}'.format(result.alternatives[0].confidence))
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
        print(u'Confidence: {}'.format(result.alternatives[0].confidence))
    
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
    response = "no" 
    diff = 0
    
    # timestamp of request
    fmt = '%Y-%m-%d %H:%M:%S.%f'
    curr_time = time.time()
    timestamp = datetime.fromtimestamp(curr_time).strftime(fmt)
     
    query_text = result.query_text
    query_intent = result.intent.display_name
    query_confidence = result.intent_detection_confidence
    query_fulfillment = result.fulfillment_text
    
    # convert to json to access parameters
    jsonObj = MessageToJson(result)
    data = json.loads(jsonObj)
    query_verb = ""
    query_object = ""
    if (data['parameters']):
        if (data['parameters']['verb']):
            query_verb = data['parameters']['verb'][0]
        if (data['parameters']['object']):
            query_object = data['parameters']['object'][0]
    
    # unsure why event.question is different
    if (query_intent == 'event.question'):
        if (data['parameters']['verb']):
            query_verb = data['parameters']['verb']
        if (data['parameters']['object']):
            query_object = data['parameters']['object']
    
    # get synonyms of entity as when questioning, it may be in a different tense
    #command = 'curl -X GET "https://api.dialogflow.com/v1/entities/verb" -H "Authorization: Bearer 22e0c5207d7d4db996ce38bb6a24d8af" -H "Content-Type: application/json"'
    #process = subprocess.Popen(command.split())
    
    print('=' * 20)
    print('Query text: {}'.format(query_text))
    print('Detected intent: {} (confidence: {})\n'.format(
        query_intent,
        query_confidence))
    print('Verb: {}'.format(query_verb))
    print('Object: {}'.format(query_object))
    print('Fulfillment text: {}\n'.format(query_fulfillment))
    print('=' * 20)
       
    # TODO: if the intent is not clear, ask user to rephrase to ensure accuracy?
    # TODO: what type of Google database system to use?
    #       need one good for relational database, SQL
    
    # for now, using a local text file to store and query from
    n=0
    if (query_intent == 'event.detect'):
        filename_entry = 'entry/' + str(n) +'.txt'
        if os.path.isfile(filename_entry):
            n+=1
            
        f = open(filename_entry, 'w+')
        f.write(query_text)
        f.close()
        filename_time = 'timestamp/' + str(n) +'.txt'
        f = open(filename_time, 'w+')
        f.write(timestamp)
        f.close()
        response = query_fulfillment
    elif (query_intent == 'event.question'):
        for filename in os.listdir('entry'):
            if filename.endswith('.txt'): 
                filename = os.path.join('entry', filename)
                f = open(filename, 'r')
                if query_verb in f.read():
                    f.close()
                    f = open(filename, 'r')
                    if query_object in f.read():
                        response = "yes"
                        f.close()
                        time_file = os.path.basename(os.path.normpath(filename))

                        f = open('timestamp/' + time_file, 'r')
                        event_time = f.read().rstrip()
                        query_time = datetime.fromtimestamp(time.time()).strftime(fmt)
                        diff = datetime.strptime(query_time, fmt)\
                               - datetime.strptime(event_time, fmt)
                        minutes = diff.seconds / 60
                        print diff.seconds
                        response = "<speak>" + response + ", you did it" + str(minutes) + " minutes ago" + "</speak>"
                        break
            
    return response
    
def synthesize_ssml(ssml):
    """Synthesizes speech from the input string of ssml.

    Note: ssml must be well-formed according to:
        https://www.w3.org/TR/speech-synthesis/

    Example: <speak>Hello there.</speak>
    """
    client = texttospeech.TextToSpeechClient()

    input_text = texttospeech.types.SynthesisInput(ssml=ssml)

    # note: the voice can also be specified by name
    # names of voices can be retrieved with client.list_voices()
    voice = texttospeech.types.VoiceSelectionParams(
        language_code='en-US',
        ssml_gender=texttospeech.enums.SsmlVoiceGender.FEMALE)

    audio_config = texttospeech.types.AudioConfig(
        audio_encoding=texttospeech.enums.AudioEncoding.MP3)

    response = client.synthesize_speech(input_text, voice, audio_config)

    # the response's audio_content is binary
    with open('output.mp3', 'wb') as out:
        out.write(response.audio_content)
        print('Audio content written to file "output.mp3"')


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
    

    engine = create_engine('mysql+pymysql://postgres:sfs@/postgres?unix_socket=/cloudsql/speaker-for-seniors:australia-southeast1:speaker-for-seniors-instance/.s.PGSQL.5432')
        
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
    parse = detect_intent_texts(project_id, args.session_id, text, 
        args.language_code)
        
    synthesize_ssml(sort_query(parse))
    command = 'totem --play output.mp3'
    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)

    

