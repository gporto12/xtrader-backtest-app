import os
import requests
import json
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd

# Importa as funções do nosso motor de backtest
from backtest_logic import buscar_dados_api, rodar_backtest

# Cria a aplicação Flask
app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# Pega a chave da API das variáveis de ambiente do Render
POLYGON_API_KEY = os.environ.get('POLYGON_API_KEY', 'SUA_CHAVE_AQUI_PARA_TESTE_LOCAL')

def create_analysis_prompt(metricas, trades):
    """Cria o prompt para a API Gemini com base nos resultados do backtest."""
    trades_str = ""
    for trade in trades[:10]:
        trades_str += f"- Trade em {pd.to_datetime(trade['data_entrada']).strftime('%d/%m/%Y')}: {trade['resultado']}, P&L: $ {float(trade['pnl_financeiro']):.2f}\n"

    prompt = f"""
    Você é um analista de risco e estrategista de trading profissional. Sua tarefa é analisar os resultados de um backtest de uma estratégia de trading e fornecer um feedback conciso, profissional e útil em português do Brasil.

    **Contexto da Estratégia:**
    A estratégia testada é a "Invert 50", uma estratégia de compra/venda baseada em médias móveis exponenciais. A entrada ocorre em um pullback para a MME 50.

    **Resultados do Backtest:**
    - **Métricas Gerais:**
        - Total de Operações: {metricas['totalOperacoes']}
        - Taxa de Acerto: {metricas['taxaAcerto']}
        - Lucro/Prejuízo Bruto Total: {metricas['lucroBrutoTotal']}
        - Drawdown Máximo: {metricas['drawdownMaximo']}
    - **Amostra de Trades:**
    {trades_str}

    **Sua Análise:**
    Com base nos dados acima, forneça uma análise em 3 partes, usando Markdown simples:
    1.  **### Diagnóstico Geral:** Faça um resumo do desempenho da estratégia.
    2.  **### Pontos Fortes:** Identifique os aspectos positivos dos resultados.
    3.  **### Pontos de Melhoria e Riscos:** Aponte as fraquezas e os riscos evidentes e sugira otimizações.

    Seja direto, objetivo e use uma linguagem que um trader entenderia.
    """
    return prompt

@app.route('/')
def home():
    """Serve a página principal do nosso site."""
    return send_from_directory('.', 'index.html')

@app.route('/backtest', methods=['POST'])
def handle_backtest_request():
    """Endpoint da API que executa o backtest."""
    try:
        data = request.get_json()
        ativo = data.get('ativo')
        data_inicio = data.get('data_inicio')
        data_fim = data.get('data_fim')
        lotes = int(data.get('lotes', 1))
        valor_ponto = float(data.get('valor_ponto', 1.0))
        estrategia = data.get('estrategia', 'venda')

        if not all([ativo, data_inicio, data_fim]):
            return jsonify({"error": "Parâmetros 'ativo', 'data_inicio' e 'data_fim' são obrigatórios."}), 400

        # CORREÇÃO: O timeframe para dados diários na API da Polygon é 'day'.
        timeframe = 'day'
        dados_historicos = buscar_dados_api(ativo, timeframe, data_inicio, data_fim, POLYGON_API_KEY)
        
        if dados_historicos is None or dados_historicos.empty:
            return jsonify({"error": "Não foi possível obter os dados históricos. Verifique o ticker do ativo e o período."}), 404

        resultados_df = rodar_backtest(dados_historicos, ativo, lotes, valor_ponto, estrategia)
        
        if resultados_df is None or resultados_df.empty:
            return jsonify({"error": "Nenhum trade foi gerado pela estratégia neste período."}), 200

        total_trades = len(resultados_df)
        taxa_acerto = (len(resultados_df[resultados_df['resultado'] == 'Gain']) / total_trades) * 100 if total_trades > 0 else 0
        lucro_bruto = resultados_df['pnl_financeiro'].sum()
        pico_anterior = resultados_df['equity_curve'].cummax()
        drawdown_financeiro = (pico_anterior - resultados_df['equity_curve']).max()
        metricas = {
            "totalOperacoes": total_trades,
            "taxaAcerto": f"{taxa_acerto:.2f}%",
            "lucroBrutoTotal": f"$ {lucro_bruto:,.2f}",
            "drawdownMaximo": f"$ {drawdown_financeiro:,.2f}"
        }
        
        historico_formatado = [{"time": int(index.timestamp()), "open": row['open'], "high": row['high'], "low": row['low'], "close": row['close']} for index, row in dados_historicos.iterrows()]
        
        resultados_df['data_entrada'] = pd.to_datetime(resultados_df['data_entrada']).astype(str)
        resultados_df['data_saida'] = pd.to_datetime(resultados_df['data_saida']).astype(str)
        trades_detalhados = resultados_df.to_dict(orient='records')

        return jsonify({"metricas": metricas, "trades": trades_detalhados, "historico_ohlc": historico_formatado})

    except Exception as e:
        print(f"Erro no servidor /backtest: {e}")
        return jsonify({"error": f"Ocorreu um erro interno no servidor: {e}"}), 500

@app.route('/analyze-results', methods=['POST'])
def analyze_results_with_gemini():
    """Endpoint que chama a API Gemini para analisar os resultados."""
    try:
        backtest_data = request.get_json()
        prompt = create_analysis_prompt(backtest_data.get('metricas'), backtest_data.get('trades'))
        
        gemini_api_key = os.environ.get('GEMINI_API_KEY', '')
        gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={gemini_api_key}"
        
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        response = requests.post(gemini_url, json=payload)
        response.raise_for_status()
        
        result = response.json()
        analysis_text = result['candidates'][0]['content']['parts'][0]['text']
        return jsonify({"analysis": analysis_text})

    except Exception as e:
        print(f"Erro na análise da IA: {e}")
        return jsonify({"error": f"Ocorreu um erro interno na análise da IA: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)
