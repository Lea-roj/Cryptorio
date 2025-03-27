from helper_functions import *


if __name__ == "__main__":
    texts = [
        """A South Korean court has temporarily lifted a three-month business suspension imposed on crypto exchange 
        Upbit, allowing the platform to resume onboarding new clients while a legal dispute with the country’s 
        Financial Intelligence Unit (FIU) remains ongoing. The FIU initially sanctioned Upbit on February 25, 
        prohibiting it from processing deposits and withdrawals for new users. The regulator claimed Upbit had 
        violated policies restricting transactions with unregistered virtual asset service providers (VASPs), 
        a breach of South Korea’s crypto compliance framework. South Korean Court Delays Upbit Suspension Pending 
        Final Ruling: In response, Upbit’s parent company, Dunamu, filed a lawsuit to overturn the suspension and 
        requested an injunction to pause the order. On March 27, local outlet Newsis reported that the court granted 
        the injunction, delaying enforcement of the suspension until 30 days after a final court ruling. The move 
        enables Upbit to continue accepting new customer registrations in the meantime. Upbit, founded in 2017, 
        is South Korea’s largest cryptocurrency exchange by trading volume. However, it has been under regulatory 
        scrutiny in recent months. In October 2023, the Financial Services Commission (FSC) began investigating the 
        exchange for potential violations of anti-monopoly laws. In a separate case, the FIU flagged Upbit for 
        possible Know Your Customer (KYC) failures, identifying as many as 600,000 potential KYC violations in a 
        review tied to the platform’s business license renewal. South Korean law mandates that crypto exchanges 
        comply with strict KYC rules following the ban on anonymous trading introduced in 2018. The FIU also accused 
        Upbit of conducting over 45,000 transactions with unregistered foreign exchanges, a violation of the Act on 
        Reporting and Using Specified Financial Transaction Information. In a broader crackdown on unlicensed 
        exchanges, South Korea’s government has increased oversight of cross-border digital asset activity. The 
        country recently mandated that businesses report crypto-related transactions used for tax evasion or currency 
        manipulation. Last week, South Korean prosecutors launched a formal 
        investigation into Bithumb, one of the country’s largest cryptocurrency exchanges, over allegations that 
        company funds were misused to facilitate an apartment purchase for its former CEO. The Seoul Southern 
        District Prosecutors’ Office also executed a search and seizure operation at Bithumb’s headquarters in 
        Yeoksam-dong. Authorities suspect that Bithumb provided a 3 billion Korean won (approximately $2.4 million) 
        lease deposit for an apartment in Seongsu-dong to its former CEO and current advisor, Kim Dae-sik. Last year, 
        South Korea’s cryptocurrency investors crossed 15 million. According to figures submitted by the Bank of 
        Korea, 15.59 million South Koreans held accounts on the nation’s top five cryptocurrency exchanges by the end 
        of November. Deposits in crypto exchanges also doubled, rising from 4.7 trillion won ($3.2 billion) in 
        October to 8.8 trillion won ($6.03 billion) in November.""",
        """Bitcoin has recently surged past the $70,000 mark, reaching an all-time high and reigniting investor 
        optimism in the cryptocurrency market. Analysts from major financial institutions, including Goldman Sachs 
        and Fidelity, have revised their forecasts, predicting Bitcoin could hit $100,000 by the end of the year. 
        This rally is fueled by a combination of growing institutional interest, the recent approval of multiple 
        Bitcoin spot ETFs in the United States, and a broader shift toward digital assets in global portfolios. The 
        upcoming Bitcoin halving event, expected in april, is also seen as a major catalyst, historically linked to 
        price increases due to reduced supply. Crypto adoption continues to rise globally, with El Salvador recently 
        announcing the launch of a Bitcoin-powered sovereign wealth fund and several European banks integrating 
        blockchain-based settlement systems. Major tech companies, including Tesla and Square, have increased their 
        Bitcoin holdings, signaling confidence in the long-term potential of decentralized finance. Experts agree 
        that the current market sentiment is overwhelmingly bullish, and retail investors are returning in droves. 
        “We’re witnessing a new wave of crypto enthusiasm, reminiscent of the 2021 bull run,” said Sarah Kim, 
        a senior analyst at CryptoMarkets. “Unlike past cycles, this one appears to be driven more by fundamentals 
        than hype.” With inflation concerns still looming and traditional markets facing uncertainty, many investors 
        see Bitcoin as a hedge and a long-term store of value. The momentum behind Bitcoin suggests that now could be 
        a strategic entry point for both new and experienced investors looking to capitalize on the next phase of 
        crypto growth."""
    ]

    results = []

    for i, text in enumerate(texts):
        result = analyze_text(text)
        result["id"] = f"doc_{i + 1}"
        results.append(result)

    with open("analyzed_articles.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)