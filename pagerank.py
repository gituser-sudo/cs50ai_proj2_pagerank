import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    trans_prob = {}

    no_of_pages = len(corpus.keys())

    common_prob = (1 - damping_factor) / no_of_pages

    if (len(corpus[page]) == 0):
        common_prob = 1 / no_of_pages

    for key in corpus.keys():
        trans_prob[key] = common_prob

    no_of_links = len(corpus[page])
    for link in corpus[page]:
        page_prob = (damping_factor) / no_of_links
        trans_prob[link] = trans_prob[link] + page_prob

    return trans_prob


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    page_count = {}

    no_of_pages = len(corpus.keys())
    for key in corpus.keys():
        page_count[key] = 0

    rand = random.random()
    sel_index = int(rand * no_of_pages)
    page = list(corpus.keys())[sel_index]

    page_count[page] = page_count[page] + 1

    # starting from page

    for i in range(n):
        rand = random.random()
        if rand < damping_factor:
            no_of_links = len(corpus[page])
            if no_of_links == 0:
                rand = random.random()
                sel_index = int(rand * no_of_pages)
                page = list(corpus.keys())[sel_index]
                page_count[page] = page_count[page] + 1
            else:
                sel_index = int(random.random() * no_of_links)
                page = list(corpus[page])[sel_index]
                page_count[page] = page_count[page] + 1
        else:
            rand = random.random()
            sel_index = int(rand * no_of_pages)
            page = list(corpus.keys())[sel_index]
            page_count[page] = page_count[page] + 1

    for page in page_count.keys():
        page_count[page] = page_count[page] / (n + 1)

    return page_count


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    page_rank = {}
    new_page_rank = {}

    no_of_pages = len(corpus.keys())
    for key in corpus.keys():
        page_rank[key] = 1 / no_of_pages

    keep_going = True
    while (keep_going):
        for key in corpus.keys():
            link_contrib = 0
            if(len(corpus[key]) == 0):
                new_page_rank[key] = 1 / no_of_pages
            else:
                for page in corpus[key]:
                    link_contrib = link_contrib + damping_factor * page_rank[page] / len(corpus[key])
                new_page_rank[key] = (1 - damping_factor) / no_of_pages + link_contrib

        keep_going = False
        # check if they add up to 1
        sum_prob = 0

        for key in corpus.keys():
            sum_prob = sum_prob + new_page_rank[key]
        print(sum_prob)

        for key in corpus.keys():
            if new_page_rank[key] - page_rank[key] > 0.001:
                keep_going = True

    return new_page_rank

if __name__ == "__main__":
    main()
