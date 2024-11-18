import React, { useState } from 'react';
import './TranslationApp.css';
import axios from 'axios';

const TranslationApp = () => {
  const [inputText, setInputText] = useState('');
  const [translatedText, setTranslatedText] = useState('');

  const handleInputChange = (e) => {
    setInputText(e.target.value);
  };

  const handleTranslate = async () => {
    try {
      const response = await axios.post('http://127.0.0.1:5000/api/translate', {
        text: inputText,
      });
      setTranslatedText(response.data.translatedText);
    } catch (error) {
      console.error('Error translating text:', error);
      setTranslatedText('Translation failed.');
    }
  };

  return (
    <div className="container">
      <h1 className="title">Text Translator</h1>
      <div className="translateBox">
        <div className="textAreaWrapper">
          <textarea
            placeholder="Enter text"
            value={inputText}
            onChange={handleInputChange}
            className="textArea"
          />
        </div>
        <div className="middleSection">
          <button onClick={handleTranslate} className="translateButton">
            Translate
          </button>
        </div>
        <div className="textAreaWrapper">
          <textarea
            placeholder="Translation"
            value={translatedText}
            readOnly
            className="textArea"
          />
        </div>
      </div>
    </div>
  );
};

export default TranslationApp;
